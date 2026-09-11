"""Three frozen-model, frame-synchronized videos with episode-local top30% intersect A>0 colors.

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
OUT=ROOT/'video-top30_positive-episode11-three-models';OUT.mkdir(exist_ok=True)
LABELS={'mixed1000':'MIXED DEMO + HIL | STEP 1000','mixed1500':'MIXED DEMO + HIL | STEP 1500','demo1500':'DEMO ONLY | STEP 1500'}

def render(spec):
    key=spec['key'];result=json.loads((OUT/f'{key}_timeline.json').read_text());n=len(result['value_common_scale']);T=n-1
    target=OUT/f'{key}_episode11_top30_positive_selected_only.mp4';validation=OUT/f'{key}_validation.json'
    if target.exists():
        assert validation.exists() and json.loads(validation.read_text())['status']=='ok'
        return str(target)
    part=target.with_suffix('.part.mp4');assert not part.exists(),'Preserve partial video'
    v=np.array(result['value_common_scale']);a=np.array(result['advantage_full']);selected=np.array(result['selected_top30_positive'])
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
    p.text((790,577),'RED: TOP30% & A>0',font=f16,fill=red);p.text((1080,577),'WHITE: REST',font=f16,fill=white)
    cut_encoder=av.open(str(part),'w');cut_stream=cut_encoder.add_stream('libx264',rate=FPS)
    cut_stream.width=W;cut_stream.height=H;cut_stream.pix_fmt='yuv420p';cut_stream.options={'crf':'20','preset':'fast','threads':'2'};cut_stream.time_base=Fraction(1,FPS);cut_count=0
    preview=max(0,events[0]-20) if events else n//2
    preview_saved=False
    frames=zip(*(camera_frames(Path(path),np.array(queries)) for path,queries in spec['camera_specs']))
    for i,(main,left,right) in enumerate(frames):
        if not selected[i]:continue
        canvas=chart.copy();canvas.paste(Image.fromarray(main).resize((640,480)),(320,54))
        canvas.paste(Image.fromarray(left).resize((320,240)),(0,174));canvas.paste(Image.fromarray(right).resize((320,240)),(960,174));d=ImageDraw.Draw(canvas)
        for xy,label in [((18,18),'LEFT WRIST'),((553,18),'MAIN VIEW'),((1080,18),'RIGHT WRIST')]:d.text(xy,label,font=f20,fill=white)
        d.text((18,440),'HELD-OUT EP 11',font=f16,fill=muted);d.text((18,467),key.upper(),font=f16,fill=white)
        d.text((980,438),f'{i/FPS:6.2f} / {T/FPS:.2f}s',font=f16,fill=white)
        d.text((980,465),'HUMAN CONTROL' if mask[i] else 'AUTONOMOUS',font=f16,fill=(255,188,80) if mask[i] else muted)
        d.text((980,492),'SELECTED: TOP30% & A>0' if selected[i] else ('TERMINAL' if i==T else 'NOT SELECTED'),font=f14,fill=red if selected[i] else white)
        d.rectangle((320,498,960,534),fill=bg)
        d.text((345,503),f'Value {v[i]:+.4f}',font=f24,fill=white);d.text((632,503),f'A {a[i]:+.4f}',font=f24,fill=red if selected[i] else white)
        d.text((320,540),LABELS[key],font=f20,fill=white)
        for py,y0,y1 in tracks:
            d.line((px[i],y0,px[i],y1),fill=cyan,width=2);d.ellipse((px[i]-3,py[i]-3,px[i]+3,py[i]+3),fill=red if selected[i] else white)
        d.text((35,926),f'Per-video TOP30% & A>0 | C=150, Z=15337 | {result["selected_count"]}/{T} nonterminal frames | offline / future data',font=f14,fill=muted)
        if selected[i]:
            d.rectangle((0,0,1280,48),fill=bg)
            d.text((25,12),'TOP30% & A>0 SELECTED ONLY | CUTS / DISCONTINUOUS | ORIGINAL TIME AT RIGHT',font=f20,fill=red)
            if not preview_saved and i>=preview:
                canvas.save(OUT/f'{key}_preview.png');preview_saved=True
            cf=av.VideoFrame.from_image(canvas);cf.pts=cut_count;cf.time_base=Fraction(1,FPS);cut_count+=1
            for packet in cut_stream.encode(cf):cut_encoder.mux(packet)
        if cut_count%400==0:print('RENDER',key,cut_count,int(selected.sum()),flush=True)
    for packet in cut_stream.encode():cut_encoder.mux(packet)
    cut_encoder.close()
    cut_decoded=0
    with av.open(str(part)) as con:
        for fr in con.decode(video=0):
            assert abs(float(fr.pts*fr.time_base)-cut_decoded/FPS)<1e-5
            cut_decoded+=1
    assert cut_count==cut_decoded==int(selected.sum())
    os.replace(part,target)
    put(validation,{'status':'ok','frames':cut_decoded,'fps':FPS,'seconds':cut_decoded/FPS,'bytes':target.stat().st_size,'full_decode_passed':True,'source_frames':n,'eligible_frames':T,'selected_count':int(selected.sum()),'selected_min_advantage':float(a[selected].min()),'color_rule':'episode-local highest30% intersect A>0; nonterminal frames including human control; stable ties by frame','value_limits':spec['value_limits'],'advantage_limits':spec['advantage_limits']})
    print('VIDEO_COMPLETE',target,flush=True);return str(target)

def main():
    source=ROOT/'video-bottom30-episode11-three-models'
    manifest=json.loads((source/'comparison_manifest.json').read_text())
    summaries=[]
    for spec in manifest['render_specs']:
        key=spec['key'];d=json.loads((source/f'{key}_timeline.json').read_text())
        a=np.asarray(d['advantage_full']);T=len(a)-1;k=math.ceil(.30*T)
        chosen=np.argsort(-a[:T],kind='stable')[:k];chosen=chosen[a[chosen]>0]
        selected=np.zeros(len(a),dtype=bool);selected[chosen]=True
        assert len(chosen)>0 and np.all(a[selected]>0)
        d.pop('selected_bottom30',None)
        d.update(selected_top30_positive=selected.tolist(),selected_count=int(selected.sum()),selection_cap=k,selected_negative_count=0,selection_min_advantage=float(a[selected].min()),selection_max_advantage=float(a[selected].max()),selection='episode-local highest30% of all nonterminal frames intersect A>0; includes human control; stable ties by original frame')
        put(OUT/f'{key}_timeline.json',d)
        edges=np.diff(np.r_[False,selected,False].astype(int));starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
        segments=[{'start_frame':int(lo),'end_frame_exclusive':int(hi),'start_s':float(lo/FPS),'end_s_exclusive':float(hi/FPS),'frames':int(hi-lo)} for lo,hi in zip(starts,stops)]
        put(OUT/f'{key}_selected_segments.json',{'segments':segments,'selected_frame_indices':np.flatnonzero(selected).tolist(),'segment_count':len(segments)})
        summary={'model':key,'eligible_frames':T,'selected_count':len(chosen),'seconds':len(chosen)/FPS,'minimum_advantage':float(a[selected].min()),'segment_count':len(segments)}
        summaries.append(summary);print('SELECTION',summary,flush=True)
    manifest.update(red_rule='highest30% intersect A>0, per model per episode',summaries=summaries)
    put(OUT/'comparison_manifest.json',manifest)
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:videos=list(pool.map(render,manifest['render_specs']))
    put(OUT/'complete.json',{'status':'ok','videos':videos,'summaries':summaries})
    print('ALL_THREE_VIDEOS_COMPLETE',flush=True)

if __name__=='__main__':main()
