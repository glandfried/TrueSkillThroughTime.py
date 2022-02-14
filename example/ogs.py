import unittest
#sudo pip3 install trueskillthroughtime
import trueskillthroughtime as ttt
from importlib import reload  # Python 3.4+ only.
reload(ttt)
import math
import pandas as pd
import time
import datetime

df = pd.read_csv('input/summary_filtered.csv')

prior_dict = dict()
for h_key in set([(h,w) for h, w in zip(df.handicap, df.width)]):
    prior_dict[str(h_key)] = ttt.Player(ttt.Gaussian(0.,25.0/3.),0.,1.0/100)
results = [[0,1] if bw==1 else [1,0] for bw in df.black_win]
composition = [[[str(w)],[str(b)]] if h<2 else [[str(w)],[str(b),str((h,s))]] for w,b,h,s in zip(df.white,df.black,df.handicap,df.width) ]   
times = [] 

print(datetime.datetime.now().time())
h = ttt.History(composition, results, times , prior_dict, mu=0.0, sigma=10.,beta=1.,gamma=0.2125,iterations=16)
print(datetime.datetime.now().time())
print("Converging History")
ts_log_evidence = h.log_evidence()
print("TS:", ts_log_evidence )
h.convergence(True)
print(datetime.datetime.now().time())
ttt_log_evidence = h.log_evidence()
print("TS: ", ts_log_evidence, ", TTT:", ttt_log_evidence)

class tests(unittest.TestCase):
    def evidence(self):
        self.assertEqual(ts_log_evidence < ttt_log_evidence, True)
        self.assertAlmostEqual(ttt_log_evidence, -195388.7125823722)

w_mean = [ t.posterior(str(w)).mu for t,w in zip(h.batches,df.white) ]  
b_mean = [ t.posterior(str(b)).mu for t,b in zip(h.batches,df.black) ]  
w_std = [ t.posterior(str(w)).sigma for t,w in zip(h.batches,df.white) ]
b_std = [ t.posterior(str(b)).sigma for t,b in zip(h.batches,df.black) ]
h_mean = [ t.posterior[str((hp,w))].mu if hp > 1 else 0 for hp,w in zip(df.handicap,df.width) ]
h_std = [ t.posteriors[str((hp,w))].sigma if hp > 1 else 0 for hp,w in zip(df.handicap,df.width) ] 
evidence = [ t.events[0].evidence for t in h.batches ] 

res = df[['id']].copy() 
res["w_mean"] = w_mean
res["w_std"] = w_std
res["b_mean"] = b_mean
res["b_std"] = b_std
res["h_mean"] = h_mean
res["h_std"] = h_std
res["evidence"] = evidence

res.to_csv("output/longtest_output.csv", index=False)

tests.evidence()
