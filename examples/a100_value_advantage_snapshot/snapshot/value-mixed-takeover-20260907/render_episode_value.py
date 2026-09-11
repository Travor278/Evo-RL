"""Read-only checkpoint inference and synchronized three-camera diagnostic video."""
import os,sys,json,time
from pathlib import Path
from fractions import Fraction
import numpy as np
import av,torch
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,'/data/experiments/value-vision-schedules-20260907')
from lean_model import LeanVisionValue
from types import SimpleNamespace
from mixed_data import ROOT,Pool,sha,put

EP=11; STEP=1000; FPS=30; HORIZON=50
RUN=ROOT/'runs/piperx-vision-mixed1to1-c150-z15337-fixed1500-seed1000'
OUT=ROOT/'video-review-episode11-step1000'; OUT.mkdir(exist_ok=True)
TARGET=OUT/'episode11_step1000_value_advantage.mp4'

def camera_frames(path,queries):
    with av.open(str(path)) as con:
        st=con.streams.video[0];st.thread_type='AUTO'
        con.seek(max(0,round(float(queries[0])/st.time_base)-1),stream=st,backward=True)
        decoder=iter(con.decode(st));prev=None;current=next(decoder)
        for q in queries:
            while float(current.pts*st.time_base)<q:
                prev=current
                nxt=next(decoder,None)
                if nxt is None:break
                current=nxt
            candidates=[x for x in [prev,current] if x is not None]
            best=min(candidates,key=lambda x:abs(float(x.pts*st.time_base)-q))
            err=abs(float(best.pts*st.time_base)-q)
            assert err<1e-4,(path,q,err)
            yield best.to_ndarray(format='rgb24')

def main():
    assert not TARGET.exists(), 'Do not overwrite completed artifact'
    m=json.loads((ROOT/'manifest.json').read_text());assert EP in m['hil']['splits']['test']['episode_indices']
    selected=json.loads(json.dumps(m));selected['hil']['splits']['test']['episode_indices']=[EP]
    d=Pool(selected,'hil','test');row=d.rows[EP];n=int(row['length']);T=n-1
    camera_specs=[]
    for key in d.cameras:
        pre='videos/'+key
        path=d.root/d.info['video_path'].format(video_key=key,chunk_index=row[pre+'/chunk_index'],file_index=row[pre+'/file_index'])
        queries=float(row[pre+'/from_timestamp'])+d.timestamps[EP].astype(np.float64)
        camera_specs.append((path,queries))
    def frames():return zip(*(camera_frames(*x) for x in camera_specs))
    predpath=OUT/'per_frame_predictions.json'
    ckpt=RUN/f'checkpoint-{STEP:06d}.pt'
    if predpath.exists():
        result=json.loads(predpath.read_text());assert result['step']==STEP and result['episode']==EP
        values=np.array(result['predicted_value']);assert len(values)==n
    else:
        torch.set_num_threads(4);device=torch.device('cuda:0')
        model=LeanVisionValue(SimpleNamespace(use_gradient_checkpointing=False)).to(device)
        saved=torch.load(ckpt,map_location='cpu',weights_only=False)
        assert saved['step']==STEP and saved['protocol']['manifest_sha256']==sha(ROOT/'manifest.json')
        model.load_state_dict(saved['model'],strict=True);del saved;model.eval()
        centers=torch.linspace(-1,0,201,device=device);pred=[];batch=[]
        with torch.inference_mode():
            for i,images in enumerate(frames()):
                batch.append(torch.from_numpy(np.stack(images)).permute(0,3,1,2))
                if len(batch)==8 or i==T:
                    x=torch.stack(batch).to(device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(images=x,image_attention_mask=torch.ones(x.shape[:2],device=device,dtype=torch.bool))
                    pred.extend((logits.float().softmax(-1)*centers).sum(-1).cpu().tolist());batch=[]
                    if (i+1)%400==0 or i==T:print('INFERENCE',i+1,n,flush=True)
        values=np.array(pred);assert len(values)==n and np.isfinite(values).all()
        result={'episode':EP,'step':STEP,'fps':FPS,'predicted_value':values.tolist(),'checkpoint':str(ckpt),'selection':'first held-out episode by index, not selected by performance','sampling':'every original frame, no interpolation','manifest_sha256':sha(ROOT/'manifest.json')}
        put(predpath,result);del model;torch.cuda.empty_cache()
    ends=np.minimum(np.arange(n)+HORIZON,T)
    boot=values.copy();boot[-1]=0
    delta=-(ends-np.arange(n))/d.scale+boot[ends]-values
    count=np.r_[0,np.cumsum(d.events[EP].astype(int))]
    penalty=-150*(count[ends]-count[np.arange(n)])/d.scale
    advantage=delta+penalty;advantage[-1]=0
    result.update(advantage_full=advantage.tolist(),time_value_delta=delta.tolist(),explicit_penalty=penalty.tolist(),human_control=d.masks[EP].tolist(),takeover_onset_frames=(np.flatnonzero(d.events[EP])+1).tolist(),horizon=HORIZON,scale=d.scale,terminal_note='Value curve is raw prediction including terminal. Advantage uses terminal bootstrap 0; terminal A=0.',color_definition='segment t to t+1: red if full A[t]>0 else white',scope='All episode frames, including human-control intervals; no autonomous-only restriction for playback.')
    put(predpath,result)
    # Check agreement with previously evaluated checkpoint on available exact frames.
    old=json.loads((RUN/f'advantage-{STEP:06d}-records.json').read_text())['records']
    errors=[abs(values[r['frame']]-r['value_before']) for r in old if r['episode']==EP and r['horizon']==50]
    assert errors and max(errors)<.002, max(errors)
    W,H=1280,820;bg=(13,17,24);white=(242,243,245);red=(255,65,76);muted=(157,169,184)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    def font(size):return ImageFont.truetype(fontpath,size)
    f16,f20,f25=font(16),font(20),font(25)
    chart=Image.new('RGB',(W,H),bg);dr=ImageDraw.Draw(chart)
    x0,x1,y0,y1=72,1230,607,755
    low=float(np.floor((values.min()-.01)*20)/20);high=min(0.,float(np.ceil((values.max()+.01)*20)/20))
    if high<=low:high=low+.05
    px=np.linspace(x0,x1,n);py=y1-(values-low)/(high-low)*(y1-y0)
    for tick in np.linspace(low,high,4):
        yy=y1-(tick-low)/(high-low)*(y1-y0);dr.line((x0,yy,x1,yy),fill=(45,53,65));dr.text((6,yy-8),f'{tick:.2f}',font=f16,fill=muted)
    for t in np.linspace(0,T/FPS,6):
        xx=x0+t*FPS/T*(x1-x0);dr.text((xx-15,y1+7),f'{t:.0f}s',font=f16,fill=muted)
    for i in range(T):dr.line((px[i],py[i],px[i+1],py[i+1]),fill=red if advantage[i]>0 else white,width=2)
    dr.text((72,563),'PREDICTED VALUE  |  full episode timeline',font=f20,fill=white)
    dr.text((735,568),'RED: A > 0',font=f16,fill=red);dr.text((920,568),'WHITE: A <= 0',font=f16,fill=white)
    # Subtle amber timeline marks show recorded takeover onsets, not predictions.
    for onset in result['takeover_onset_frames']:
        xx=px[onset];dr.line((xx,592,xx,602),fill=(255,188,80),width=2)
    part=OUT/'episode11_step1000_value_advantage.part.mp4'
    assert not part.exists(),'Preserve existing partial artifact; choose a new name for recovery'
    encoder=av.open(str(part),'w');stream=encoder.add_stream('libx264',rate=FPS)
    stream.width=W;stream.height=H;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast','threads':'4'}
    stream.time_base=Fraction(1,FPS)
    preview_frame=max(0,result['takeover_onset_frames'][0]-20) if result['takeover_onset_frames'] else n//2
    for i,(main,left,right) in enumerate(frames()):
        canvas=chart.copy();canvas.paste(Image.fromarray(main).resize((640,480)),(320,54))
        canvas.paste(Image.fromarray(left).resize((320,240)),(0,174));canvas.paste(Image.fromarray(right).resize((320,240)),(960,174))
        p=ImageDraw.Draw(canvas)
        p.text((18,18),'LEFT WRIST',font=f20,fill=white);p.text((553,18),'MAIN VIEW',font=f20,fill=white);p.text((1080,18),'RIGHT WRIST',font=f20,fill=white)
        p.text((18,440),f'HELD-OUT EP {EP}',font=f16,fill=muted);p.text((18,468),f'Checkpoint {STEP}',font=f16,fill=muted)
        p.text((985,440),f'{i/FPS:6.2f} / {T/FPS:.2f} s',font=f16,fill=white)
        p.text((985,468),'HUMAN CONTROL' if d.masks[EP][i] else 'AUTONOMOUS',font=f16,fill=(255,188,80) if d.masks[EP][i] else muted)
        p.rectangle((320,498,960,534),fill=bg)
        p.text((345,503),f'Value {values[i]:+.4f}',font=f25,fill=white)
        p.text((630,503),f'Advantage {advantage[i]:+.4f}',font=f25,fill=red if advantage[i]>0 else white)
        p.line((px[i],y0,px[i],y1),fill=(71,207,240),width=2);p.ellipse((px[i]-4,py[i]-4,px[i]+4,py[i]+4),fill=red if advantage[i]>0 else white)
        p.text((45,793),'A: 50-frame future window, includes recorded takeover penalty. Offline diagnostic. 30 fps / 1x.',font=f16,fill=muted)
        if i==preview_frame:canvas.save(OUT/'preview.png')
        fr=av.VideoFrame.from_image(canvas);fr.pts=i;fr.time_base=Fraction(1,FPS)
        for packet in stream.encode(fr):encoder.mux(packet)
        if (i+1)%600==0:print('RENDER',i+1,n,flush=True)
    for packet in stream.encode():encoder.mux(packet)
    encoder.close()
    with av.open(str(part)) as con:
        decoded=0
        for fr in con.decode(video=0):
            assert abs(float(fr.pts*fr.time_base)-decoded/FPS)<1e-5
            decoded+=1
        assert decoded==n,(decoded,n)
    os.replace(part,TARGET)
    put(OUT/'validation.json',{'status':'ok','frames':n,'fps':FPS,'seconds':n/FPS,'bytes':TARGET.stat().st_size,'video':str(TARGET),'max_value_difference_vs_prior_batched_inference':max(errors),'strict_source_pts_tolerance':1e-4,'full_output_decode_passed':True})
    print('VIDEO_COMPLETE',TARGET,flush=True)

if __name__=='__main__':main()
