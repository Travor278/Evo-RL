"""Select raw-A frames only; no expansion, smoothing, padding, or new inference."""
import json, math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_demo_heldout_extremes import writer, encode, finish, camera_frames, put, LABELS

ROOT = Path('/data/experiments/value-mixed-takeover-20260907')
SOURCE = ROOT / 'video-demo-heldout-mixed-predictions'
OUT = ROOT / 'video-demo-heldout-raw10-selected-frames'
FPS = 30

def render(spec):
    key, group = spec['key'], spec['group']
    name = spec.get('name', key + '_' + group)
    out = Path(spec.get('out', OUT))
    data = json.loads(Path(spec.get('timeline', SOURCE / f'{key}_timeline.json')).read_text())
    a, v = np.array(data['advantage_full']), np.array(data['value_common_scale'])
    assert np.isfinite(a).all() and np.isfinite(v).all()
    n = len(a)
    k = math.ceil((n - 1) * .1)
    positive = group.startswith('top')
    if 'selected_frames' in spec:
        selected = np.array(spec['selected_frames'], dtype=int)
        k = len(selected)
    else:
        selected = np.argsort(-a[:-1] if positive else a[:-1], kind='stable')[:k]
        if positive:
            selected = selected[a[selected] > 0]
    selected = np.sort(selected)
    assert len(selected) and np.all(selected < n-1)
    mask = np.zeros(n, dtype=bool)
    mask[selected] = True
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    label = spec.get('label', 'TOP10% & A>0' if positive else 'BOTTOM10%')
    scope = spec.get('scope', 'within this episode')
    summary = dict(key=name, episode=data['episode'], checkpoint=data['checkpoint'],
                   eligible_nonterminal_frames=n-1, requested_count=k,
                   selected_count=len(selected), output_source_frame_indices=selected.tolist(),
                   selection=f'{scope}; stable raw advantage rank; top also A>0',
                   intervals=[dict(start=int(s), end_exclusive=int(e)) for s,e in zip(starts,stops)],
                   min_selected_A=float(a[selected].min()), max_selected_A=float(a[selected].max()),
                   playback='Selected frames only, chronological, 30fps/1x; gaps jump immediately; no 50-frame expansion',
                   metric='Existing raw advantage unchanged (50-transition estimator); this is NOT a new one-step advantage',
                   source_root=data['root'])
    if 'global_threshold' in spec: summary['global_threshold'] = spec['global_threshold']
    put(out / f'{name}_selection.json', summary)

    bg, white, red, muted, cyan = (13,17,24), (242,243,245), (255,65,76), (157,169,184), (71,207,240)
    fontpath = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    f14, f16, f20, f24 = [ImageFont.truetype(fontpath, s) for s in (14,16,20,24)]
    chart = Image.new('RGB', (1280,960), bg)
    d = ImageDraw.Draw(chart)
    px = np.linspace(74,1230,n)
    tracks = []
    for curve, limits, ys, title in [(v,spec['value_limits'],(605,722),'PREDICTED VALUE'),
                                     (a,spec['advantage_limits'],(793,887),'RAW ADVANTAGE | CURRENT FRAME')]:
        low, high = limits
        y0,y1 = ys
        py = y1-(curve-low)/(high-low)*(y1-y0)
        tracks.append((py,y0,y1))
        d.text((74,y0-31),title,font=f20,fill=white)
        for tick in np.linspace(low,high,3):
            yy=y1-(tick-low)/(high-low)*(y1-y0)
            d.line((74,yy,1230,yy),fill=(45,53,65))
            d.text((4,yy-8),f'{tick:.3f}',font=f14,fill=muted)
        d.line(list(zip(px,py)),fill=white,width=2)
        # Exact selected points only: never mark a nonselected neighbouring frame red.
        for i in selected:
            d.ellipse((px[i]-1,py[i]-1,px[i]+1,py[i]+1),fill=red)
    for t in np.linspace(0,(n-1)/FPS,6):
        d.text((74+t*FPS/(n-1)*1156-15,898),f'{t:.0f}s',font=f16,fill=muted)
    d.text((645,577),'RED: SELECTED FRAMES | WHITE: NOT SELECTED',font=f14,fill=red)
    d.text((74,744),'NO WINDOW EXPANSION | SOURCE GAPS ARE SKIPPED',font=f14,fill=muted)

    w = writer(out / f'{name}_raw_selected.mp4')
    cameras = [camera_frames(Path(path), np.array(q)[selected]) for path,q in spec['camera_specs']]
    count=0
    for j, (i, images) in enumerate(zip(selected,zip(*cameras))):
        main,left,right=images
        canvas=chart.copy()
        canvas.paste(Image.fromarray(main).resize((640,480)),(320,54))
        canvas.paste(Image.fromarray(left).resize((320,240)),(0,174))
        canvas.paste(Image.fromarray(right).resize((320,240)),(960,174))
        d=ImageDraw.Draw(canvas)
        d.text((18,12),label+' | RAW-A SELECTED FRAMES ONLY | NO 50-FRAME EXPANSION',font=f20,fill=red)
        d.text((18,440),f'DEMO TEST EP {data["episode"]}',font=f16,fill=muted)
        d.text((18,467),key.upper(),font=f16,fill=white)
        d.text((18,495),'NOT POLICY TRAINING',font=f14,fill=muted)
        d.text((980,438),f'Original {i/FPS:.2f}s',font=f16,fill=white)
        d.text((980,465),f'Source frame {i}',font=f16,fill=white)
        d.text((980,492),f'Selected {j+1}/{len(selected)}',font=f16,fill=red)
        d.rectangle((320,498,960,534),fill=bg)
        d.text((333,503),f'Current V {v[i]:+.4f}',font=f24,fill=white)
        d.text((650,503),f'Current A {a[i]:+.4f}',font=f24,fill=red)
        d.text((320,540),LABELS[key],font=f20,fill=white)
        for py,y0,y1 in tracks:
            d.line((px[i],y0,px[i],y1),fill=cyan,width=2)
            d.ellipse((px[i]-3,py[i]-3,px[i]+3,py[i]+3),fill=cyan)
        d.text((25,928),f'30fps / 1x | {label} {scope} | Existing raw A unchanged',font=f14,fill=muted)
        encode(w,canvas,j)
        count+=1
        if j==len(selected)//2: canvas.save(out/f'{name}_preview.png')
    assert count==len(selected)
    summary['video']=finish(w,len(selected))
    put(out/f'{name}_validation.json',summary)
    print('RAW_VIDEO_COMPLETE', name, count, summary['min_selected_A'], summary['max_selected_A'],flush=True)
    return summary

def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((SOURCE/'comparison_manifest.json').read_text())
    jobs=[dict(spec,group=g) for spec in manifest['render_specs'] for g in ['top10','bottom10']]
    assert len(jobs)==4
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(render,jobs))
    put(OUT/'complete.json',dict(status='ok',results=results))
    print('ALL_RAW_VIDEOS_COMPLETE',flush=True)

if __name__=='__main__': main()
