"""Freeze labels/splits and fail closed on overlap, scale or decoding problems."""
import argparse,json
import numpy as np
from mixed_data import ROOT,Pool,make_manifest,put,rewards,BalancedBatches,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--metadata-only',action='store_true');args=ap.parse_args()
    r,g,b=rewards([0,0,1,1,0,0]);assert np.array_equal(r,[-1,-151,-1,-1,-1,0])
    assert g[0]==-155 and b.sum()==1
    assert rewards([1,1,1,0,0])[2].sum()==0
    assert rewards([0,0,1])[2].sum()==0
    for rank in range(4):
        batches=list(BalancedBatches([100,200],8,rank,4,0,10))
        assert all(sum(i<100 for i in batch)==4 for batch in batches)
        assert batches==list(BalancedBatches([100,200],8,rank,4,0,10))
        assert batches[5:]==list(BalancedBatches([100,200],8,rank,4,5,10))
    m=make_manifest();pools={};summaries={};seen={};overlaps=[]
    for source in ['demo','hil']:
        for split in ['train','test']:
            d=Pool(m,source,split);pools[source,split]=d
            for e,h in d.fingerprints.items():
                if h in seen:overlaps.append([seen[h],[source,split,e]])
                seen[h]=[source,split,e]
            summaries[source+'_'+split]=dict(episodes=len(d.ids),frames=len(d),max_cost=max(-g[0] for g in d.returns.values()),takeovers=sum(int(b.sum()) for b in d.events.values()))
    assert not overlaps,('Duplicate trajectory action/state fingerprints require review',overlaps)
    maxcost=max(summaries[s+'_train']['max_cost'] for s in ['demo','hil'])
    assert maxcost==15337 and m['scale']==maxcost
    cfg=json.loads((ROOT/'reward_config.json').read_text());assert cfg['normalization_multiplier']==1 and cfg['expected_return_scale']==maxcost
    if (ROOT/'manifest.json').exists():assert json.loads((ROOT/'manifest.json').read_text())==m
    else:put(ROOT/'manifest.json',m)
    report=dict(status='metadata_ok',summaries=summaries,scale=maxcost,exact_action_state_duplicate_pairs=overlaps,split_leakage_note='Exact whole-trajectory fingerprint check only; similar scenes/session correlation still possible',unit_tests='reward boundaries and deterministic 1:1 sampler passed',manifest_sha256=sha(ROOT/'manifest.json'))
    put(ROOT/'metadata_preflight.json',report)
    print(json.dumps(report),flush=True)
    if args.metadata_only:return
    transfer=json.loads((ROOT/'transfer_verified.json').read_text());assert transfer['status']=='ok'
    decoded=0
    # Verify newly transferred HIL at first/middle/last of every episode, all cameras.
    for split in ['train','test']:
        d=pools['hil',split]
        for start,n in zip(d.starts,d.lengths):
            for frame in [0,int(n)//2,int(n)-1]:
                sample=d[int(start)+frame];assert tuple(sample['images'].shape)==(3,3,480,640);decoded+=3
    report.update(status='ok',decoded_camera_frames=decoded,transfer_files=transfer['files'])
    put(ROOT/'preflight.json',report);print('PREFLIGHT_COMPLETE',flush=True)

if __name__=='__main__':main()
