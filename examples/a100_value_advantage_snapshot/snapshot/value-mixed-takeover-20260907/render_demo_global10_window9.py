"""Global configurable n-step advantage and playback union. No new inference."""
import json, math, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_demo_heldout_extremes import writer, encode, finish, camera_frames, put

ROOT=Path('/data/experiments/value-mixed-takeover-20260907')
SOURCE=ROOT/'video-demo-global30-step1500'
OUT=ROOT/'video-demo-global10-window9-step1500'
FPS=30; H=9; PREVIEW=[11,23,40]

def advantage(r,horizon):
    assert horizon>0 and r['step']==1500 and r['scale']==15337
    assert all(r['human_control']), 'Only pure teleoperation with no takeover events'
    v=np.array(r['value_common_scale'],dtype=np.float64)
    t=np.arange(len(v));end=np.minimum(t+horizon,len(v)-1)
    bootstrap=v.copy();bootstrap[-1]=0.
    a=-(end-t)/r['scale']+bootstrap[end]-v;a[-1]=0.
    assert np.isfinite(a).all()
    return a

def render(spec):
    H=int(spec.get('horizon',9));OUT=Path(spec.get('out',ROOT/'video-demo-global10-window9-step1500'))
    AH=int(spec.get('advantage_horizon',50))
    seconds=H/FPS
    ep=spec['episode'];group=spec['group'];key=f'ep{ep:03d}_mixed1500_{group}'
    r=json.loads((SOURCE/f'episode-{ep:03d}.json').read_text())
    a=advantage(r,AH);v=np.array(r['value_common_scale']);n=len(a)
    anchors=np.array(spec['anchors'],dtype=int)
    coverage=np.zeros(n,dtype=bool);latest=np.full(n,-1,dtype=int)
    for t in anchors:coverage[t:min(t+H,n)]=True;latest[t:min(t+H,n)]=t
    mapping=np.flatnonzero(coverage);assert len(mapping)>0
    edges=np.diff(np.r_[False,coverage,False].astype(int))
    summary=dict(key=key,episode=ep,checkpoint=r['checkpoint'],group=group,
                 selection_scope='All56 DEMO test episodes pooled by nonterminal frame; stable ties by episode/frame',
                 global_threshold=spec['threshold'],anchor_indices=anchors.tolist(),
                 min_anchor_A=float(a[anchors].min()),max_anchor_A=float(a[anchors].max()),
                 output_source_frame_indices=mapping.tolist(),
                 windows=[dict(anchor=int(t),end_exclusive=min(int(t)+H,n),frames_available=min(H,n-int(t)),anchor_A=float(a[t])) for t in anchors],
                 union_intervals=[dict(start=int(s),end_exclusive=int(e)) for s,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))],
                 short_tail_windows=int((anchors+H>n).sum()),
                 playback_horizon=H,playback_seconds=seconds,advantage_estimator_horizon=AH,
                 note=f'All selected anchors expanded to [t,t+{H}), overlapping windows merged. Current A need not satisfy anchor rank. Never cross episode boundary; no fabricated tail images.')
    put(OUT/f'{key}_window_manifest.json',summary)
    bg=(13,17,24);white=(242,243,245);red=(255,65,76);muted=(157,169,184);cyan=(71,207,240);yellow=(255,188,80)
    fp='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    f14,f16,f20,f24=[ImageFont.truetype(fp,s) for s in (14,16,20,24)]
    chart=Image.new('RGB',(1280,960),bg);d=ImageDraw.Draw(chart);px=np.linspace(74,1230,n);tracks=[]
    for curve,limits,ys,title in [(v,spec['value_limits'],(605,722),'PREDICTED VALUE'),(a,spec['advantage_limits'],(793,887),'RAW ADVANTAGE | CURRENT FRAME')]:
        low,high=limits;y0,y1=ys;py=y1-(curve-low)/(high-low)*(y1-y0);tracks.append((py,y0,y1))
        d.text((74,y0-31),title,font=f20,fill=white)
        for tick in np.linspace(low,high,3):
            yy=y1-(tick-low)/(high-low)*(y1-y0);d.line((74,yy,1230,yy),fill=(45,53,65));d.text((4,yy-8),f'{tick:.3f}',font=f14,fill=muted)
        for i in range(n-1):d.line((px[i],py[i],px[i+1],py[i+1]),fill=red if coverage[i] else white,width=2)
    d.text((640,577),f'RED: {seconds:.3f}s WINDOW COVERAGE, NOT CURRENT A SIGN',font=f14,fill=red)
    d.text((74,744),'YELLOW TICKS: SELECTED START FRAMES | CYAN: CURRENT FRAME',font=f14,fill=yellow)
    for t in anchors:d.line((px[t],765,px[t],769),fill=yellow)
    for t in np.linspace(0,(n-1)/FPS,6):d.text((74+t*FPS/(n-1)*1156-15,898),f'{t:.0f}s',font=f16,fill=muted)
    label='TOP10% & A>0' if group=='top10' else 'BOTTOM10%'
    w=writer(OUT/f'{key}_window{H}_union.mp4')
    cameras=[camera_frames(Path(p),np.array(q)[mapping]) for p,q in r['camera_specs']]
    count=0
    for j,(i,images) in enumerate(zip(mapping,zip(*cameras))):
        t=latest[i];main,left,right=images;canvas=chart.copy()
        canvas.paste(Image.fromarray(main).resize((640,480)),(320,54))
        canvas.paste(Image.fromarray(left).resize((320,240)),(0,174));canvas.paste(Image.fromarray(right).resize((320,240)),(960,174))
        d=ImageDraw.Draw(canvas)
        d.text((18,12),f'{label} GLOBAL | ALL {H}-STEP / {seconds:.3f}s WINDOWS | OVERLAPS MERGED',font=f20,fill=red)
        d.text((18,440),f'DEMO TEST EP {ep}',font=f16,fill=muted)
        d.text((18,467),f'MIXED1500 | A H={AH}',font=f16,fill=white)
        d.text((18,495),'NOT POLICY TRAINING',font=f14,fill=muted)
        d.text((980,438),f'Original {i/FPS:.2f}s',font=f16,fill=white)
        d.text((980,465),f'Current V {v[i]:+.4f}',font=f16,fill=white)
        d.text((980,492),f'Step {i-t+1}/{H}',font=f20,fill=red)
        d.rectangle((320,498,960,534),fill=bg)
        d.text((333,503),f'Anchor A {a[t]:+.4f}',font=f24,fill=red)
        d.text((670,503),f'Now A {a[i]:+.4f}',font=f24,fill=white)
        d.text((320,540),f'ALL56 POOLED | Anchor cutoff {spec["threshold"]:+.6f}',font=f20,fill=white)
        for py,y0,y1 in tracks:
            d.line((px[t],y0,px[t],y1),fill=yellow,width=1)
            d.line((px[i],y0,px[i],y1),fill=cyan,width=2)
            d.ellipse((px[i]-3,py[i]-3,px[i]+3,py[i]+3),fill=cyan)
        d.text((25,928),f'30fps / 1x | Latest covering start {t} | Playback H={H} | Raw advantage estimator H={AH}',font=f14,fill=muted)
        encode(w,canvas,j);count+=1
        if j==len(mapping)//2:canvas.save(OUT/f'{key}_preview.png')
    assert count==len(mapping)
    summary['video']=finish(w,len(mapping));put(OUT/f'{key}_validation.json',summary)
    print('VIDEO_COMPLETE',key,'anchors',len(anchors),'frames',len(mapping),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--horizon',type=int,default=9)
    parser.add_argument('--advantage-horizon',type=int,default=50)
    args=parser.parse_args();H=args.horizon;AH=args.advantage_horizon;assert H>0 and AH>0
    suffix=f'-adv{AH}' if AH!=50 else ''
    OUT=ROOT/f'video-demo-global10{suffix}-window{H}-step1500'
    OUT.mkdir(exist_ok=True)
    s=json.loads((SOURCE/'global_summary.json').read_text())
    rows=[json.loads((SOURCE/f'episode-{r["episode"]:03d}.json').read_text()) for r in s['per_episode']]
    assert len(rows)==56 and [r['episode'] for r in rows][:3]==PREVIEW
    assert all(r['step']==1500 and r['fps']==30 for r in rows)
    # Independently reproduce the old scores first, then compute the requested horizon.
    assert all(np.allclose(advantage(r,50),r['advantage_full'],atol=1e-12,rtol=0) for r in rows)
    arrays=[advantage(r,AH)[:-1] for r in rows];a=np.concatenate(arrays)
    assert len(a)==122970 and np.isfinite(a).all()
    k=math.ceil(.1*len(a));top=np.argsort(-a,kind='stable')[:k];top=top[a[top]>0];bottom=np.argsort(a,kind='stable')[:k]
    masks={g:np.zeros(len(a),dtype=bool) for g in ['top10','bottom10']}
    masks['top10'][top]=True;masks['bottom10'][bottom]=True
    thresholds=dict(top10=float(a[top].min()),bottom10=float(a[bottom].max()))
    jobs=[];counts=[];offset=0;reference=s['render_specs'][0]
    for r,arr in zip(rows,arrays):
        ep=r['episode'];count=dict(episode=ep,eligible_frames=len(arr))
        if AH!=50:
            put(OUT/f'ep{ep:03d}_advantage.json',dict(episode=ep,checkpoint=r['checkpoint'],value_source=str(SOURCE/f'episode-{ep:03d}.json'),horizon=AH,scale=r['scale'],gamma=1,advantage_full=np.r_[arr,0.].tolist(),formula='-min(H,T-t)/15337 + V[min(t+H,T)] - V[t]; V[T]=0 for bootstrap; terminal A=0; no takeover events'))
        for group in ['top10','bottom10']:
            anchors=np.flatnonzero(masks[group][offset:offset+len(arr)])
            count[group]=len(anchors)
            put(OUT/f'ep{ep:03d}_{group}_anchors.json',dict(episode=ep,anchors=anchors.tolist(),global_threshold=thresholds[group]))
            if ep in PREVIEW:
                jobs.append(dict(episode=ep,group=group,anchors=anchors.tolist(),threshold=thresholds[group],value_limits=reference['value_limits'],advantage_limits=reference['advantage_limits'],horizon=H,advantage_horizon=AH,out=str(OUT)))
        offset+=len(arr);counts.append(count)
    old_a=np.concatenate([np.array(r['advantage_full'])[:-1] for r in rows])
    old_top=np.argsort(-old_a,kind='stable')[:k];old_top=old_top[old_a[old_top]>0]
    old_bottom=np.argsort(old_a,kind='stable')[:k]
    overlap=dict(top_shared=int(np.intersect1d(top,old_top).size),bottom_shared=int(np.intersect1d(bottom,old_bottom).size))
    put(OUT/'global_summary.json',dict(status='ok',source=str(SOURCE),checkpoint_step=1500,episode_count=56,eligible_frames=len(a),requested_count=k,selected_top=len(top),selected_bottom=len(bottom),thresholds=thresholds,per_episode=counts,preview_episodes=PREVIEW,playback_horizon=H,advantage_horizon=AH,overlap_with_A50=overlap,global_A_min=float(a.min()),global_A_max=float(a.max()),positive_frames=int((a>0).sum()),note='Chart axes retained from A50 outputs for direct comparison'))
    assert a.min()>=reference['advantage_limits'][0] and a.max()<=reference['advantage_limits'][1]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:results=list(pool.map(render,jobs))
    put(OUT/'complete.json',dict(status='ok',results=results));print(f'ALL_WINDOW{H}_COMPLETE',flush=True)

if __name__=='__main__':main()
