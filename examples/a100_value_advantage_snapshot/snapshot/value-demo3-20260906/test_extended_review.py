import json
import numpy as np
from extended_value_review import distribution_rows, summarize_array, paired_rows

centers=np.array([-1.,-.5,0.]); y=np.array([-.5,-.5])
p=np.array([[.25,.5,.25],[.495,.01,.495]])
d=distribution_rows(p,y,centers)
np.testing.assert_allclose(d['prediction'],y)
assert -np.log(p[1,1]) > -np.log(p[0,1])
assert np.all(d['coverage90']==1)
q=distribution_rows(np.eye(3),centers,centers)
np.testing.assert_allclose(q['std'],0); np.testing.assert_allclose(q['crps'],0)
a=np.column_stack([np.array([0,1]),[0,0],y,d['prediction'],-np.log(p[:,1]),d['entropy'],d['std'],d['crps'],d['coverage90'],d['width90']])
summary=summarize_array(a,p,centers,100,30); assert summary['mae']==0
json.dumps(summary,allow_nan=False)
scale=1000.; rows=np.array([[0,0,-.1,-.09],[0,50,-.05,-.04],[1,10,-.09,-.09],[1,60,-.04,-.03]])
pairs=paired_rows(rows,{0:[0],1:[10]},scale,30)
np.testing.assert_allclose(pairs[:,2],[0,.01],atol=1e-12)
print('UNIT_OK: mean-vs-CE, quantiles, CRPS, serialization, exact 50-frame pairs')
