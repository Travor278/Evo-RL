"""Replay complete H=50 candidate action windows, never sign-filter their interiors.

This held-out episode is for diagnostics, not a policy training sampler trace.
Export the exact window union and 12 deterministic complete-window examples.
"""
import os,json,math
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import av
from PIL import Image,ImageDraw,ImageFont
from render_episode_value import camera_frames

ROOT=Path('/data/experiments/value-mixed-takeover-20260907')
SOURCE=ROOT/'video-demo-heldout-mixed-predictions'
OUT=ROOT/'video-demo-heldout-extremes5_10-chunk50'
FPS=30;HORIZON=50
LABELS={'mixed1000':'MIXED DEMO + HIL | STEP 1000','mixed1500':'MIXED DEMO + HIL | STEP 1500','demo1500':'DEMO ONLY | STEP 1500'}
def put(path,value):
    with path.open('w') as f:json.dump(value,f,indent=2)

def writer(path):
    assert not path.exists() and not path.with_suffix('.part.mp4').exists()
    part=path.with_suffix('.part.mp4');con=av.open(str(part),'w');st=con.add_stream('libx264',rate=FPS)
    st.width=1280;st.height=960;st.pix_fmt='yuv420p';st.time_base=Fraction(1,FPS)
    st.options={'crf':'20','preset':'fast','threads':'3'}
    return con,st,part,path

def encode(w,canvas,index):
    frame=av.VideoFrame.from_image(canvas);frame.pts=index;frame.time_base=Fraction(1,FPS)
    for packet in w[1].encode(frame):w[0].mux(packet)

def finish(w,expected):
    for packet in w[1].encode():w[0].mux(packet)
    w[0].close();count=0
    with av.open(str(w[2])) as con:
        for frame in con.decode(video=0):
            assert abs(float(frame.pts*frame.time_base)-count/FPS)<1e-5
            count+=1
    assert count==expected
    os.replace(w[2],w[3])
    return {'path':str(w[3]),'status':'ok','frames':count,'seconds':count/FPS,'bytes':w[3].stat().st_size,'full_decode_and_pts_ok':True}

def render(spec):
    basekey=spec['key'];group=spec['group'];key=f'{basekey}_{group}';positive=group.startswith('top');fraction=.10 if group.endswith('10') else .05;group_label=('TOP' if positive else 'BOTTOM')+str(round(100*fraction))+'%'+(' & A>0' if positive else '');data=json.loads((SOURCE/f'{basekey}_timeline.json').read_text())
    a=np.array(data['advantage_full']);v=np.array(data['value_common_scale']);human=np.array(data['human_control']);n=len(a)
    k=math.ceil(fraction*(n-1));ranked=np.argsort(-a[:-1] if positive else a[:-1],kind='stable')[:k]
    if positive:ranked=ranked[a[ranked]>0]
    anchors=np.sort(ranked);assert len(anchors)>0
    if positive:assert np.all(a[anchors]>0)
    coverage=np.zeros(n,dtype=bool);latest=np.full(n,-1,dtype=int);counts=np.zeros(n,dtype=int)
    windows=[]
    for t in anchors:
        stop=min(int(t)+HORIZON,n);coverage[t:stop]=True;latest[t:stop]=t;counts[t:stop]+=1
        window_human=human[t:stop]
        windows.append({'anchor':int(t),'end_exclusive':stop,'anchor_s':float(t/FPS),'anchor_advantage':float(a[t]),'video_frames_available':stop-int(t),'unavailable_future_video_frames':HORIZON-(stop-int(t)),'anchor_human_control':bool(human[t]),'crosses_controller_boundary':bool(np.any(window_human!=human[t]))})
    complete_anchors=anchors[anchors+HORIZON<=n]
    picks=complete_anchors[np.linspace(0,len(complete_anchors)-1,12,dtype=int)]
    needed=np.zeros(n,dtype=bool)
    for t in picks:needed[t:t+HORIZON]=True
    edges=np.diff(np.r_[False,coverage,False].astype(int));starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
    mapping=np.flatnonzero(coverage).tolist()
    put(OUT/f'{key}_window_manifest.json',{'key':key,'source_root':data['root'],'episode':11,'selection':group_label+' of all nonterminal anchors, per model; includes human control','H':50,'action_offsets':list(range(50)),'windows':windows,'union_original_frame_indices':mapping,'union_intervals':[{'start':int(lo),'end_exclusive':int(hi)} for lo,hi in zip(starts,stops)],'example_anchors':picks.tolist(),'example_selection':'12 equally spaced ranks in chronological list of selected anchors with all50 source frames; not performance-selected','note':'candidate windows on a held-out episode, not actual policy sampler records; no controller-boundary mask applied; unavailable future images not fabricated'})
    W=1280;HEIGHT=960;bg=(13,17,24);white=(242,243,245);red=(255,65,76);muted=(157,169,184);cyan=(71,207,240);yellow=(255,188,80)
    def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
    f14,f16,f20,f24=font(14),font(16),font(20),font(24)
    chart=Image.new('RGB',(W,HEIGHT),bg);p=ImageDraw.Draw(chart);x0,x1=74,1230;px=np.linspace(x0,x1,n);tracks=[]
    for curve,limits,ys,title in [(v,spec['value_limits'],(605,722),'PREDICTED VALUE'),(a,spec['advantage_limits'],(793,887),'RAW ADVANTAGE (current frame)')]:
        low,high=limits;y0,y1=ys;py=y1-(curve-low)/(high-low)*(y1-y0);tracks.append((py,y0,y1))
        p.text((74,y0-31),title,font=f20,fill=white)
        for tick in np.linspace(low,high,3):
            yy=y1-(tick-low)/(high-low)*(y1-y0);p.line((x0,yy,x1,yy),fill=(45,53,65));p.text((4,yy-8),f'{tick:.3f}',font=f14,fill=muted)
        for i in range(n-1):p.line((px[i],py[i],px[i+1],py[i+1]),fill=red if coverage[i] else white,width=2)
    for t in np.linspace(0,(n-1)/FPS,6):p.text((x0+t*FPS/(n-1)*(x1-x0)-15,895),f'{t:.0f}s',font=f16,fill=muted)
    p.text((620,577),'RED: SELECTED WINDOW COVERAGE (not current A sign)',font=f16,fill=red)
    for t in anchors:p.line((px[t],763,px[t],767),fill=yellow)
    p.text((74,744),'YELLOW TICKS: SELECTED CHUNK ANCHORS',font=f14,fill=yellow)

    def compose(images,i,t,mode,example_index=None):
        main,left,right=images;canvas=chart.copy();canvas.paste(Image.fromarray(main).resize((640,480)),(320,54))
        canvas.paste(Image.fromarray(left).resize((320,240)),(0,174));canvas.paste(Image.fromarray(right).resize((320,240)),(960,174));d=ImageDraw.Draw(canvas)
        title=group_label+' | ALL 50-STEP WINDOWS | OVERLAPS MERGED' if mode=='union' else f'{group_label} | COMPLETE CHUNK {example_index+1}/12 | FIXED ANCHOR, 50 STEPS'
        d.text((18,12),title,font=f20,fill=red)
        d.text((18,440),'DEMO TEST EP 11',font=f16,fill=muted);d.text((18,467),key.upper(),font=f16,fill=white)
        d.text((18,495),'NOT HIL / NOT POLICY TRAIN',font=f14,fill=muted)
        d.text((980,438),f'Original {i/FPS:.2f}s',font=f16,fill=white)
        d.text((980,465),'TELEOP DEMO',font=f16,fill=yellow if human[i] else muted)
        d.text((980,492),f'Step {i-t+1}/50',font=f20,fill=red)
        d.rectangle((320,498,960,534),fill=bg)
        d.text((333,503),f'Anchor A {a[t]:+.4f}',font=f24,fill=red)
        d.text((667,503),f'Now A {a[i]:+.4f}',font=f24,fill=white)
        d.text((320,540),LABELS[basekey],font=f20,fill=white)
        for py,y0,y1 in tracks:
            d.line((px[i],y0,px[i],y1),fill=cyan,width=2);d.ellipse((px[i]-3,py[i]-3,px[i]+3,py[i]+3),fill=cyan)
            d.line((px[t],y0,px[t],y1),fill=yellow,width=1)
        label='Latest covering anchor' if mode=='union' else 'Fixed sample anchor'
        d.text((25,926),f'{label}: t={t} ({t/FPS:.2f}s) | Current V={v[i]:+.4f} | 1 chunk = 50 actions | origin {group_label}',font=f14,fill=muted)
        return canvas

    w=writer(OUT/f'{key}_chunk50_union.mp4');written=0;cache={}
    decoded_sources=zip(*(camera_frames(Path(path),np.array(q)) for path,q in spec['camera_specs']))
    for i,images in enumerate(decoded_sources):
        if needed[i]:cache[i]=tuple(img.copy() for img in images)
        if not coverage[i]:continue
        canvas=compose(images,i,int(latest[i]),'union');encode(w,canvas,written);written+=1
        if written==min(600,len(mapping)):canvas.save(OUT/f'{key}_union_preview.png')
        if written%1200==0:print('UNION',key,written,len(mapping),flush=True)
    union=finish(w,len(mapping));print('UNION_COMPLETE',key,flush=True)
    ew=writer(OUT/f'{key}_positive_chunk50_examples.mp4');written=0
    for c,t in enumerate(picks):
        for i in range(int(t),int(t)+HORIZON):
            canvas=compose(cache[i],i,int(t),'example',c);encode(ew,canvas,written);written+=1
            if c==3 and i-t==25:canvas.save(OUT/f'{key}_example_preview.png')
    examples=finish(ew,len(picks)*HORIZON)
    summary={'key':key,'selected_anchors':len(anchors),'union':union,'examples':examples,'example_count':len(picks),'minimum_anchor_advantage':float(a[anchors].min()),'maximum_anchor_advantage':float(a[anchors].max()),'short_tail_windows':sum(w['video_frames_available']<50 for w in windows),'controller_crossing_windows':sum(w['crosses_controller_boundary'] for w in windows)}
    put(OUT/f'{key}_validation.json',summary);print('MODEL_COMPLETE',key,flush=True);return summary

def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((SOURCE/'comparison_manifest.json').read_text())
    jobs=[dict(spec,group=group) for group in ['top10','bottom10','top5','bottom5'] for spec in manifest['render_specs']]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:results=list(pool.map(render,jobs))
    put(OUT/'complete.json',{'status':'ok','results':results});print('ALL_CHUNK_VIDEOS_COMPLETE',flush=True)

if __name__=='__main__':main()
