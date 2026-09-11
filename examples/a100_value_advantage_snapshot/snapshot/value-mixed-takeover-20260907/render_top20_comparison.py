"""Three frozen-model, frame-synchronized videos with episode-local top20% colors.

Ranking changes only the visualization mask, not the raw advantage values.
Demo-only predictions are converted from their training scale to common cost
units before applying the identical evaluation reward and denominator.
"""
import os,sys,json,math
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import av,torch
from PIL import Image,ImageDraw,ImageFont
from safetensors.torch import load_file
sys.path.insert(0,'/data/experiments/value-vision-schedules-20260907')
from lean_model import LeanVisionValue,ASSETS
from mixed_data import ROOT,Pool,put,sha
from render_episode_value import camera_frames

EP=11;FPS=30;HZ=50;Z=15337
RUN=ROOT/'runs/piperx-vision-mixed1to1-c150-z15337-fixed1500-seed1000'
OUT=ROOT/'video-top20-episode11-three-models';OUT.mkdir(exist_ok=True)
LABELS={'mixed1000':'MIXED DEMO + HIL | STEP 1000','mixed1500':'MIXED DEMO + HIL | STEP 1500','demo1500':'DEMO ONLY | STEP 1500'}

def render(spec):
    key=spec['key'];result=json.loads((OUT/f'{key}_timeline.json').read_text());n=len(result['value_common_scale']);T=n-1
    target=OUT/f'{key}_episode11_top20.mp4';validation=OUT/f'{key}_validation.json'
    if target.exists():
        assert validation.exists() and json.loads(validation.read_text())['status']=='ok'
        return str(target)
    part=target.with_suffix('.part.mp4');assert not part.exists(),'Preserve partial video'
    v=np.array(result['value_common_scale']);a=np.array(result['advantage_full']);selected=np.array(result['selected_top20'])
    mask=np.array(result['human_control']);events=result['takeover_onset_frames']
    W,H=1280,960;bg=(13,17,24);white=(242,243,245);red=(255,65,76);muted=(157,169,184);cyan=(71,207,240)
    def font(s):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',s)
    f14,f16,f20,f24=font(14),font(16),font(20),font(24)
    chart=Image.new('RGB',(W,H),bg);p=ImageDraw.Draw(chart)
    x0,x1=74,1230;px=np.linspace(x0,x1,n)
    tracks=[]
    for data,(low,high),(y0,y1),title in [(v,spec['value_limits'],(605,722),'PREDICTED VALUE (common cost scale)'),(a,spec['advantage_limits'],(793,887),'RAW ADVANTAGE (50-frame window)')]:
        py=y1-(data-low)/(high-low)*(y1-y0);tracks.append((py,y0,y1))
        p.text((74,y0-31),title,font=f20,fill=white)
        for tick in np.linspace(low,high,3):
            yy=y1-(tick-low)/(high-low)*(y1-y0);p.line((x0,yy,x1,yy),fill=(45,53,65));p.text((4,yy-8),f'{tick:.3f}',font=f14,fill=muted)
        if low<0<high:
            yy=y1-(0-low)/(high-low)*(y1-y0);p.line((x0,yy,x1,yy),fill=(104,113,125))
        for i in range(T):p.line((px[i],py[i],px[i+1],py[i+1]),fill=red if selected[i] else white,width=2)
    for t in np.linspace(0,T/FPS,6):p.text((x0+t*FPS/T*(x1-x0)-15,895),f'{t:.0f}s',font=f16,fill=muted)
    for onset in events:p.line((px[onset],593,px[onset],602),fill=(255,188,80),width=2)
    p.text((810,577),'RED: TOP 20%',font=f16,fill=red);p.text((1010,577),'WHITE: REST',font=f16,fill=white)
    encoder=av.open(str(part),'w');stream=encoder.add_stream('libx264',rate=FPS)
    stream.width=W;stream.height=H;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast','threads':'3'}
    stream.time_base=Fraction(1,FPS)
    preview=max(0,events[0]-20) if events else n//2
    frames=zip(*(camera_frames(Path(path),np.array(queries)) for path,queries in spec['camera_specs']))
    for i,(main,left,right) in enumerate(frames):
        canvas=chart.copy();canvas.paste(Image.fromarray(main).resize((640,480)),(320,54))
        canvas.paste(Image.fromarray(left).resize((320,240)),(0,174));canvas.paste(Image.fromarray(right).resize((320,240)),(960,174));d=ImageDraw.Draw(canvas)
        for xy,label in [((18,18),'LEFT WRIST'),((553,18),'MAIN VIEW'),((1080,18),'RIGHT WRIST')]:d.text(xy,label,font=f20,fill=white)
        d.text((18,440),'HELD-OUT EP 11',font=f16,fill=muted);d.text((18,467),key.upper(),font=f16,fill=white)
        d.text((980,438),f'{i/FPS:6.2f} / {T/FPS:.2f}s',font=f16,fill=white)
        d.text((980,465),'HUMAN CONTROL' if mask[i] else 'AUTONOMOUS',font=f16,fill=(255,188,80) if mask[i] else muted)
        d.text((980,492),'TOP 20% SELECTED' if selected[i] else ('TERMINAL' if i==T else 'NOT SELECTED'),font=f14,fill=red if selected[i] else white)
        d.rectangle((320,498,960,534),fill=bg)
        d.text((345,503),f'Value {v[i]:+.4f}',font=f24,fill=white);d.text((632,503),f'A {a[i]:+.4f}',font=f24,fill=red if selected[i] else white)
        d.text((320,540),LABELS[key],font=f20,fill=white)
        for py,y0,y1 in tracks:
            d.line((px[i],y0,px[i],y1),fill=cyan,width=2);d.ellipse((px[i]-3,py[i]-3,px[i]+3,py[i]+3),fill=red if selected[i] else white)
        d.text((35,926),f'Per-video / per-model top20% | C=150, Z=15337 | {result["selected_count"]}/{T} nonterminal frames | offline / future data',font=f14,fill=muted)
        if i==preview:canvas.save(OUT/f'{key}_preview.png')
        fr=av.VideoFrame.from_image(canvas);fr.pts=i;fr.time_base=Fraction(1,FPS)
        for packet in stream.encode(fr):encoder.mux(packet)
        if (i+1)%1200==0:print('RENDER',key,i+1,n,flush=True)
    for packet in stream.encode():encoder.mux(packet)
    encoder.close()
    decoded=0
    with av.open(str(part)) as con:
        for fr in con.decode(video=0):
            assert abs(float(fr.pts*fr.time_base)-decoded/FPS)<1e-5
            decoded+=1
    assert decoded==n
    os.replace(part,target)
    put(validation,{'status':'ok','frames':n,'fps':FPS,'seconds':n/FPS,'bytes':target.stat().st_size,'full_decode_passed':True,'selected_count':int(selected.sum()),'eligible_frames':T,'selection_min_advantage':result['selection_min_advantage'],'color_rule':'top20% by descending raw full advantage, stable ties by frame; nonterminal frames including human-control segments','value_limits':spec['value_limits'],'advantage_limits':spec['advantage_limits']})
    print('VIDEO_COMPLETE',target,flush=True);return str(target)

def main():
    m=json.loads((ROOT/'manifest.json').read_text());selected=json.loads(json.dumps(m));selected['hil']['splits']['test']['episode_indices']=[EP]
    d=Pool(selected,'hil','test');row=d.rows[EP];n=int(row['length']);T=n-1;assert d.scale==Z
    cams=[]
    for key in d.cameras:
        p='videos/'+key
        path=d.root/d.info['video_path'].format(video_key=key,chunk_index=row[p+'/chunk_index'],file_index=row[p+'/file_index'])
        q=float(row[p+'/from_timestamp'])+d.timestamps[EP].astype(np.float64);cams.append((str(path),q.tolist()))
    cached=json.loads((ROOT/'video-review-episode11-step1000/per_frame_predictions.json').read_text())
    assert cached['episode']==EP and cached['step']==1000 and cached['manifest_sha256']==sha(ROOT/'manifest.json')
    predictions={'mixed1000':np.array(cached['predicted_value'])}
    configs={'mixed1000':(1000,15337,str(RUN/'checkpoint-001000.pt')),'mixed1500':(1500,15337,str(RUN/'checkpoint-001500.pt')),'demo1500':(1500,14882,str(ASSETS/'vision-only-step1500-model.safetensors'))}
    models={};torch.set_num_threads(4);device=torch.device('cuda:0')
    for key in ['mixed1500','demo1500']:
        path=OUT/f'{key}_native_predictions.json'
        if path.exists():
            cache=json.loads(path.read_text());assert cache['checkpoint']==configs[key][2] and cache['episode']==EP
            predictions[key]=np.array(cache['values']);continue
        model=LeanVisionValue(SimpleNamespace(use_gradient_checkpointing=False)).to(device)
        if key=='mixed1500':
            saved=torch.load(configs[key][2],map_location='cpu',weights_only=False);assert saved['step']==1500 and saved['protocol']['manifest_sha256']==sha(ROOT/'manifest.json')
            model.load_state_dict(saved['model'],strict=True);del saved
        else:
            export=json.loads((ASSETS/'export_validation.json').read_text());assert export['status']=='ok' and export['final_reload_strict'] and export['logits_max_abs_diff']['bfloat16']==0
            oldprotocol=json.loads((Path(export['source_checkpoint']).parent/'protocol.json').read_text())
            assert oldprotocol['variant']=='vision_only' and oldprotocol['arguments']['steps']==1500 and oldprotocol['return_scale']==14882
            model.load_state_dict(load_file(configs[key][2]),strict=True)
        models[key]=model.eval();predictions[key]=[]
    centers=torch.linspace(-1,0,201,device=device);batch=[]
    if models:
        with torch.inference_mode():
            for i,images in enumerate(zip(*(camera_frames(Path(p),np.array(q)) for p,q in cams))):
                batch.append(torch.from_numpy(np.stack(images)).permute(0,3,1,2))
                if len(batch)==8 or i==T:
                    x=torch.stack(batch).to(device)
                    for key,model in models.items():
                        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(images=x,image_attention_mask=torch.ones(x.shape[:2],device=device,dtype=torch.bool))
                        predictions[key].extend((logits.float().softmax(-1)*centers).sum(-1).cpu().tolist())
                    batch=[]
                    if (i+1)%800==0 or i==T:print('INFERENCE',list(models),i+1,n,flush=True)
        for key in models:
            predictions[key]=np.array(predictions[key]);put(OUT/f'{key}_native_predictions.json',{'episode':EP,'checkpoint':configs[key][2],'values':predictions[key].tolist()})
    del models;torch.cuda.empty_cache()
    common_values=[];common_adv=[];specs=[]
    for key,(step,training_scale,checkpoint) in configs.items():
        native=np.array(predictions[key]);assert len(native)==n and np.isfinite(native).all()
        v=native*training_scale/Z
        ends=np.minimum(np.arange(n)+HZ,T);boot=v.copy();boot[-1]=0
        count=np.r_[0,np.cumsum(d.events[EP].astype(int))]
        delta=-(ends-np.arange(n))/Z+boot[ends]-v
        penalty=-150*(count[ends]-count[np.arange(n)])/Z;a=delta+penalty;a[-1]=0
        k=math.ceil(.2*T);chosen=np.argsort(-a[:T],kind='stable')[:k]
        top=np.zeros(n,dtype=bool);top[chosen]=True;assert top.sum()==k
        put(OUT/f'{key}_timeline.json',{'episode':EP,'step':step,'checkpoint':checkpoint,'training_scale':training_scale,'display_scale':Z,'predicted_value_native':native.tolist(),'value_common_scale':v.tolist(),'advantage_full':a.tolist(),'time_value_delta':delta.tolist(),'explicit_penalty':penalty.tolist(),'selected_top20':top.tolist(),'selected_count':k,'selection_min_advantage':float(a[chosen].min()),'human_control':d.masks[EP].tolist(),'takeover_onset_frames':(np.flatnonzero(d.events[EP])+1).tolist(),'selection':'episode-local, per model; all nonterminal frames including human control; descending full A; ceil(20%); stable ties by frame index','comparison_note':'Same physical episode, input cameras, timestamps, evaluation C150/Z15337/n50 and terminal bootstrap0. Demo-only trained on time cost without HIL; mixed trained on time+takeover costs. Scale conversion does not erase target/data/training differences. Demo old8x4 vs mixed4x8, global32.'})
        common_values.extend(v.tolist());common_adv.extend(a.tolist());specs.append({'key':key,'camera_specs':cams})
    vlo=math.floor((min(common_values)-.01)*20)/20;vhi=min(0.,math.ceil((max(common_values)+.01)*20)/20)
    amp=math.ceil(max(abs(min(common_adv)),abs(max(common_adv)))*100)/100+.005
    for spec in specs:spec.update(value_limits=[vlo,vhi],advantage_limits=[-amp,amp])
    put(OUT/'comparison_manifest.json',{'episode':EP,'frames':n,'fps':FPS,'models':configs,'shared_value_limits':[vlo,vhi],'shared_advantage_limits':[-amp,amp],'red_rule':'within episode each model top20%, NOT A>0','render_specs':specs})
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:videos=list(pool.map(render,specs))
    put(OUT/'complete.json',{'status':'ok','videos':videos,'frames_each':n,'fps':FPS})
    print('ALL_THREE_VIDEOS_COMPLETE',flush=True)

if __name__=='__main__':main()
