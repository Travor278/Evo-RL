"""Read-only dataset review; output only separate review artifacts."""
import json, math, collections
from pathlib import Path
from fractions import Fraction
import numpy as np
import pyarrow.parquet as pq
import av
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/data/experiments/value-mixed-takeover-20260907')
DST=Path('/data/datasets/piperx-demo558-value1500-a50-top10-union20-v1')
OUT=ROOT/'review-demo558-final-v1'
OUT.mkdir(exist_ok=True)
def put(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
validation=json.loads((DST/'selection/validation.json').read_text())
assert validation['status']=='ok'
assert json.loads((ROOT/'demo558-top10-a50-w20-build/release/complete.json').read_text())==validation
clips=json.loads((DST/'selection/segments.json').read_text())
meta={int(r['episode_index']):r for f in (DST/'meta/episodes').rglob('*.parquet') for r in pq.read_table(f).to_pylist()}
info=json.loads((DST/'meta/info.json').read_text())
assert len(meta)==len(clips)==7146
for c in clips:assert c['length']==meta[c['episode_index']]['length']
length=np.array([c['length'] for c in clips]); assert int(length.sum())==322261
plan=json.loads((ROOT/'demo558-top10-a50-w20-build/release/plan.json').read_text())
source_info=json.loads((Path(plan['source'])/'meta/info.json').read_text())
source_meta={int(r['episode_index']):r for f in (Path(plan['source'])/'meta/episodes').rglob('*.parquet') for r in pq.read_table(f).to_pylist()}
by_source=collections.defaultdict(list)
for c in clips:by_source[c['source_episode']].append(c)
sources=[]
for ep,m in sorted(source_meta.items()):
    cc=by_source[ep]; sources.append(dict(episode=ep,original_frames=m['length'],segments=len(cc),retained_frames=sum(c['length'] for c in cc),anchors=sum(len(c['anchor_frames']) for c in cc)))
bins=[0,20,30,60,90,150,300,math.inf]
labels=['<20','20-29','30-59','60-89','90-149','150-299','>=300']
summary=dict(validation=validation,quantiles_frames={str(q):float(np.percentile(length,q)) for q in [0,25,50,75,90,95,99,100]},mean_frames=float(length.mean()),duration_bins=[dict(label=l,count=int(((length>=a)&(length<b)).sum())) for a,b,l in zip(bins,bins[1:],labels)],coverage=length.sum()/1255729,shorter_than20=int((length<20).sum()),exact20=int((length==20).sum()),sources_without_clips=[s['episode'] for s in sources if s['segments']==0],split={split:dict(segments=sum(c['source_value_model_split']==split for c in clips),frames=sum(c['length'] for c in clips if c['source_value_model_split']==split),sources=len({c['source_episode'] for c in clips if c['source_value_model_split']==split})) for split in ['train','test']},files={str(p.relative_to(DST)):dict(count=sum(f.is_file() for f in p.rglob('*')),bytes=sum(f.stat().st_size for f in p.rglob('*') if f.is_file())) for p in DST.iterdir() if p.is_dir()})
put('summary.json',summary)
put('distribution_data.json',dict(clips=[{k:v for k,v in c.items() if k not in ['anchor_frames','anchor_advantages']}|dict(anchor_count=len(c['anchor_frames']),start_advantage=c['anchor_advantages'][0]) for c in clips],sources=sources,anchor_advantages=[v for c in clips for v in c['anchor_advantages']],anchor_relative_positions=[t/source_meta[c['source_episode']]['length'] for c in clips for t in c['anchor_frames']]))
put('actual_features.json',info['features'])
rng=np.random.default_rng(1000); used=set(); chosen=[]
for group,indices in enumerate(np.array_split(np.argsort(length,kind='stable'),4)):
    for i in rng.permutation(indices):
        c=clips[int(i)]
        if c['source_episode'] in used:continue
        used.add(c['source_episode']);chosen.append(dict(c,preview_group=group+1))
        if len(chosen)==(group+1)*5:break
assert len(chosen)==20 and len(used)==20
put('examples20.json',chosen)
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
F=ImageFont.truetype(FONT,23);SM=ImageFont.truetype(FONT,19);BIG=ImageFont.truetype(FONT,34)
W,H=1280,720; FPS=30
class Writer:
    def __init__(self,path):
        self.con=av.open(str(path),'w',options={'movflags':'+faststart'});self.st=self.con.add_stream('libx264',rate=FPS);self.st.width=W;self.st.height=H;self.st.pix_fmt='yuv420p';self.st.options={'crf':'22','preset':'fast','threads':'2'};self.i=0
    def add(self,img):
        f=av.VideoFrame.from_image(img);f.pts=self.i;f.time_base=Fraction(1,FPS);self.i+=1
        for p in self.st.encode(f):self.con.mux(p)
    def close(self):
        for p in self.st.encode():self.con.mux(p)
        self.con.close()
def decoded(path,start,n):
    with av.open(str(path)) as con:
        st=con.streams.video[0];con.seek(int(start/FPS/st.time_base),stream=st,backward=True)
        count=0
        for f in con.decode(st):
            idx=round(float(f.pts*f.time_base)*FPS)
            if idx<start:continue
            if idx>=start+n:break
            assert idx==start+count;count+=1;yield f.to_image()
        assert count==n,(path,count,n)
cams=['observation.images.camera_wrist_left','observation.images.camera_top','observation.images.camera_wrist_right']
master=Writer(OUT/'examples20_compilation.mp4');thumbs=[];qa=[]
for number,c in enumerate(chosen,1):
    ep=c['episode_index'];m=meta[ep];n=c['length'];gens=[]
    for cam in cams:
        pre='videos/'+cam; path=DST/info['video_path'].format(video_key=cam,chunk_index=m[pre+'/chunk_index'],file_index=m[pre+'/file_index'])
        start=round(m[pre+'/from_timestamp']*FPS);assert start==c['video_from_index'];gens.append(decoded(path,start,n))
    pred=json.loads((ROOT/'demo558-top10-a50-w20-build/predictions'/f"episode-{c['source_episode']:03d}.json").read_text())
    anchors=set(c['anchor_frames']); name=f'example{number:02d}_ep{ep:05d}.mp4';writer=Writer(OUT/name)
    for i,images in enumerate(zip(*gens)):
        img=Image.new('RGB',(W,H),'#0d1622');draw=ImageDraw.Draw(img);srcframe=c['source_from']+i
        draw.text((20,14),f"Example {number:02d}/20 | Export EP {ep} | Source EP {c['source_episode']} | Length group {c['preview_group']}/4",font=F,fill='white')
        draw.text((20,49),f"Complete segment: {n} frames / {n/FPS:.2f}s | Original [{c['source_from']/FPS:.2f}, {c['source_to']/FPS:.2f})s | Value {c['source_value_model_split']} source",font=SM,fill='#bcc9d6')
        img.paste(images[0].resize((320,240)),(0,250));img.paste(images[1].resize((640,480)),(320,110));img.paste(images[2].resize((320,240)),(960,250))
        draw.text((20,218),'LEFT WRIST',font=SM,fill='white');draw.text((565,83),'MAIN VIEW',font=SM,fill='white');draw.text((987,218),'RIGHT WRIST',font=SM,fill='white')
        draw.text((20,602),f"Start A50 {c['anchor_advantages'][0]:+.5f} | Now A50 {pred['advantage_full'][srcframe]:+.5f} | Frame {i+1}/{n} | 1x",font=F,fill='white')
        color='#ff766b' if srcframe in anchors else '#bcc9d6'
        draw.text((20,640),'CURRENT FRAME: '+('TOP10% ANCHOR' if srcframe in anchors else 'RETAINED WINDOW FRAME (not an anchor)'),font=SM,fill=color)
        for j in range(n):
            col='#ff766b' if c['source_from']+j in anchors else '#7f909f';x=20+int(j*1240/n);x2=20+int((j+1)*1240/n);draw.rectangle((x,681,x2,692),fill=col)
        x=20+int(i*1240/n);draw.line((x,675,x,699),fill='#55ddec',width=3)
        writer.add(img);master.add(img)
        if i==n//2:
            img.save(OUT/f'example{number:02d}.jpg',quality=88);thumbs.append(img.resize((480,270)))
    writer.close()
    # One explicit inter-episode title card in the compilation only; independent clips have no pauses.
    card=Image.new('RGB',(W,H),'#0d1622');d=ImageDraw.Draw(card);d.text((170,310),f'END EXAMPLE {number:02d} | {n/FPS:.2f}s complete segment',font=BIG,fill='white')
    for _ in range(30):master.add(card)
    with av.open(str(OUT/name)) as con:
        count=sum(1 for _ in con.decode(video=0));assert count==n
    qa.append(dict(file=name,frames=n,duration=n/FPS));print('PREVIEW_COMPLETE',number,ep,n,flush=True)
master.close()
sheet=Image.new('RGB',(1920,1350),'#0d1622')
for i,img in enumerate(thumbs):sheet.paste(img,((i%4)*480,(i//4)*270))
sheet.save(OUT/'contact20.jpg',quality=90)
with av.open(str(OUT/'examples20_compilation.mp4')) as con:
    count=sum(1 for _ in con.decode(video=0));assert count==sum(c['length'] for c in chosen)+600
put('media_validation.json',dict(status='ok',videos=qa,compilation_frames=count,selection_seed=1000,selection='Five per length rank quartile; distinct source episodes. Not population-frequency weighted.'))
print('REVIEW_MEDIA_COMPLETE',json.dumps(summary),flush=True)
