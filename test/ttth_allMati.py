#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:12:14 2020

@author: mati
"""
import os
name = os.path.basename(__file__).split(".py")[0]
import pandas as pd
import sys
sys.path.append('/home/mati/Storage/Tesis/AnalisisGo-Tesis/')
#import ipdb
import TTTorg as th
from importlib import reload  # Python 3.4+ only.
reload(th)
env = th.TrueSkill(draw_probability=0,tau=1,beta=4.33,epsilon=0.1)
largo = 300
df = pd.read_csv('/home/mati/Storage/Tesis/AnalisisGo-Tesis/DatosPurificados/summary_filtered_handicapPositive.csv')
df=df[largo:largo*2]

from collections import defaultdict
prior_dict = defaultdict(lambda:env.Rating(0,25/3,0,1/100))
for h_key in set([(h,s) for h, s in zip(df.handicap, df.width) ]):
    prior_dict[h_key] 
baches = []
bache = 1
count = 0
#%%
for i in range(largo):
    baches.append(bache)
    count += 1
    if count>100:
        count = 0
        bache += 1
    
results = list(df.black_win.map(lambda x: [1,0] if x else [0,1] ) )
composition = [ [[w],[b]] if h<2 else [[w],[b,(h,s)]] for w, b, h, s in zip(df.white, df.black, df.handicap, df.width) ]   

#%%
history = env.history(games_composition=composition,batch_numbers=None, results=results, prior_dict=prior_dict)
history.through_time(online=False)
history.convergence()
# %%
w_mean = [ t.posteriors[w].mu for t,w in zip(history.times,df.white) ]    
#%%                                                        
b_mean = [ t.posteriors[b].mu for t,b in zip(history.times,df.black) ]                                                            
w_std = [ t.posteriors[w].sigma for t,w in zip(history.times,df.white)]                                                          
b_std = [ t.posteriors[b].sigma for t,b in zip(history.times,df.black) ]     
h_mean = [  t.posteriors[(h,w)].mu if h > 1 else 0 for t,h,w in zip(history.times,df.handicap,df.width) ]
h_std = [  t.posteriors[(h,w)].sigma if h > 1 else 0 for t,h,w in zip(history.times,df.handicap,df.width) ] 
evidence = [  t.evidence[0] for t in history.times] 
last_evidence = [  t.last_evidence[0] for t in history.times] 

# %%
res = df[['id']].copy() 
res["w_mean"] = w_mean
res["w_std"] = w_std
res["b_mean"] = b_mean
res["b_std"] = b_std
res["h_mean"] = h_mean
res["h_std"] = h_std
res["evidence"] = evidence
res["last_evidence"] = last_evidence

res.to_csv(name+".csv", index=False)
#%%
for i in range(len(w_mean)):
    w_mean[i] = round(w_mean[i],10)
    w_mean2[i] = round(w_mean2[i],10)
    last_evidence[i] = round(last_evidence[i],10)
    last_evidence2[i] = round(last_evidence2[i],10)
    