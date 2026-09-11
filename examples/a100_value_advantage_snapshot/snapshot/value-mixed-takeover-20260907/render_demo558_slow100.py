import json
from pathlib import Path
from fractions import Fraction
import numpy as np
import pyarrow.parquet as pq
import av
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/data/experiments/value-mixed-takeover-20260907')
DST=Path('/data/datasets/piperx-demo558-value1500-a50-top10-union20-v1')
OUT=ROOT/'review-demo558-slow100-v1';OUT.mkdir(exist_ok=True)
assert json.loads((DST/'selection/validation.json').read_text())['status']=='ok'
cs=json.loads((DST/'selection/segments.json').read_text())
meta={int(r['episode_index']):r for f in (DST/'meta/episodes').rglob('*.parquet') for r in pq.read_table(f).to_pylist()}
info=json.loads((DST/'meta/info.json').read_text())
assert all(c['length']==meta[c['episode_index']]['length'] for c in cs)
N=len(cs);F=sum(c['length'] for c in cs);short=[c for c in cs if c['length']<50]
full=sum(max(c['length']-49,0) for c in cs);anchors=sum(len(c['anchor_frames']) for c in cs)
va=sum(sum(t+50<=c['source_to'] for t in c['anchor_frames']) for c in cs)
stats=dict(total_episodes=N,total_frames=F,short_episodes=len(short),short_episode_pct=len(short)/N*100,short_frames=sum(c['length'] for c in short),short_frame_pct=sum(c['length'] for c in short)/F*100,atleast50=N-len(short),exact50=sum(c['length']==50 for c in cs),valid_full50_starts=full,requires_padding_starts=F-full,requires_padding_pct=(F-full)/F*100,selected_anchors=anchors,full50_selected_anchors=va,selected_anchors_needing_padding=anchors-va,definition='50 action timesteps include current timestep, never cross new episode boundary. This is length feasibility, not a corrupt-data diagnosis.')
(OUT/'chunk50_statistics.json').write_text(json.dumps(stats,indent=2))
rng=np.random.default_rng(1000)
chosen=[cs[int(i)] for i in rng.choice(N,100,replace=False)]
total=sum(c['length'] for c in chosen)
offset=0;manifest=[]
for c in chosen:
    manifest.append(dict(c,playback_from_seconds=offset/15,playback_to_seconds=(offset+c['length'])/15));offset+=c['length']
(OUT/'examples100.json').write_text(json.dumps(manifest,indent=2))
W,H=1280,650;FPS=15
fnt='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fnt,23);small=ImageFont.truetype(fnt,19)
def frames(path,start,n):
    with av.open(str(path)) as con:
        st=con.streams.video[0];con.seek(int(start/30/st.time_base),stream=st,backward=True);count=0
        for f in con.decode(st):
            idx=round(float(f.pts*f.time_base)*30)
            if idx<start:continue
            if idx>=start+n:break
            assert idx==start+count;count+=1;yield f.to_image()
        assert count==n
path=OUT/'examples100_half_speed_no_cuts.mp4'
assert not path.exists(),'Keep prior artifact intact'
con=av.open(str(path),'w',options={'movflags':'+faststart'});st=con.add_stream('libx264',rate=FPS);st.width=W;st.height=H;st.pix_fmt='yuv420p';st.options={'crf':'22','preset':'fast','threads':'4'}
index=0;cams=['observation.images.camera_wrist_left','observation.images.camera_top','observation.images.camera_wrist_right']
for number,c in enumerate(chosen,1):
    ep=c['episode_index'];n=c['length'];m=meta[ep];gens=[]
    for cam in cams:
        pre='videos/'+cam;p=DST/info['video_path'].format(video_key=cam,chunk_index=m[pre+'/chunk_index'],file_index=m[pre+'/file_index'])
        start=round(m[pre+'/from_timestamp']*30);assert start==c['video_from_index'];gens.append(frames(p,start,n))
    for i,images in enumerate(zip(*gens)):
        img=Image.new('RGB',(W,H),'#0d1622');draw=ImageDraw.Draw(img)
        draw.text((15,12),f"{number:03d}/100 | EP {ep} | Source EP {c['source_episode']} | {n} frames ({n/30:.2f}s original) | 0.5x speed",font=font,fill='white')
        img.paste(images[0].resize((320,240)),(0,208));img.paste(images[1].resize((640,480)),(320,88));img.paste(images[2].resize((320,240)),(960,208))
        draw.text((18,179),'LEFT WRIST',font=small,fill='white');draw.text((566,58),'MAIN',font=small,fill='white');draw.text((985,179),'RIGHT WRIST',font=small,fill='white')
        draw.text((18,583),f"Original {c['source_from']/30:.2f}-{c['source_to']/30:.2f}s | Start A50 {c['anchor_advantages'][0]:+.5f} | Frame {i+1}/{n}",font=font,fill='#ced8e0')
        draw.rectangle((18,626,1262,632),fill='#425260');draw.rectangle((18,626,18+int((i+1)/n*1244),632),fill='#55ddec')
        f=av.VideoFrame.from_image(img);f.pts=index;f.time_base=Fraction(1,FPS);index+=1
        for packet in st.encode(f):con.mux(packet)
        if number in [1,25,50,75,100] and i==n//2:img.save(OUT/f'preview{number:03d}.jpg',quality=90)
    print('RENDERED',number,ep,n,flush=True)
for packet in st.encode():con.mux(packet)
con.close();assert index==total
with av.open(str(path)) as con:
    st=con.streams.video[0];assert float(st.average_rate)==15
    count=0
    for f in con.decode(st):
        assert abs(float(f.pts*f.time_base)-count/15)<1e-5;count+=1
    assert count==total
qa=dict(status='ok',episodes=100,frames=count,duration_seconds=count/15,original_duration_seconds=count/30,playback_speed=.5,transition_frames=0,source='actual exported dataset',seed=1000,sampling='uniform without replacement among all 7146 segments',bytes=path.stat().st_size)
(OUT/'validation.json').write_text(json.dumps(qa,indent=2));print('DONE',json.dumps(qa),flush=True)
