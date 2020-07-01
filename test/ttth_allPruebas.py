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
sys.path.append('/home/mati/Storage/Tesis/AnalisisGo-Tesis/trueskill.py/')
#import ipdb
import src as th
import numpy as np
from importlib import reload  # Python 3.4+ only.
reload(th)
env = th.TrueSkill(draw_probability=0,tau=1,beta=4.33,epsilon=0.1)
largo = 3000
df = pd.read_csv('/home/mati/Storage/Tesis/AnalisisGo-Tesis/DatosPurificados/summary_filtered_handicapPositive.csv')
df['year'] = df['started'].apply(lambda row: int(row[0:4]))
df['date'] = df['started'].apply(lambda row: row[0:7])


df=df[0:20]

from collections import defaultdict
prior_dict = defaultdict(lambda:env.Rating(0,25/3,0,1/100))
for h_key in set([(h,s) for h, s in zip(df.handicap, df.width) ]):
    prior_dict[h_key] 
baches = []
bache = 1
count = 0
#%%
for i in range(len(df.date)-1):

    if (df.date[i]!=df.date[i+1]) & (i<len(df.date)):
        bache += 1
    baches.append(bache)
        
    
results = list(df.black_win.map(lambda x: [1,0] if x else [0,1] ) )
composition = [ [[w],[b]] if h<2 else [[w],[b,(h,s)]] for w, b, h, s in zip(df.white, df.black, df.handicap, df.width) ]   

#%%
history = env.history(games_composition=composition,batch_numbers=baches, results=results, prior_dict=prior_dict)
history.through_time(online=False)
history.convergence()
# %%
w_mean = [t.posteriors[w].mu for t,w in zip(history.times,df.white)]                                                          
b_mean = [t.posteriors[b].mu for t,b in zip(history.times,df.black)]                                                            
w_std = [t.posteriors[w].sigma for t,w in zip(history.times,df.white)]                                                          
b_std = [t.posteriors[b].sigma for t,b in zip(history.times,df.black)]     
h_mean = [t.posteriors[(h,w)].mu if h > 1 else 0 for t,h,w in zip(history.times,df.handicap,df.width) ]
h_std = [t.posteriors[(h,w)].sigma if h > 1 else 0 for t,h,w in zip(history.times,df.handicap,df.width) ] 
evidence = [t.evidence[0] for t in history.times] 
last_evidence = [t.last_evidence[0] for t in history.times] 

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
    
    
variables 1.430511474609375e-06
v y w 5.0067901611328125e-06
funcion 9.5367431640625e-06
variables 1.6689300537109375e-06
v y w 5.245208740234375e-06
funcion 9.298324584960938e-06

variables 1.430511474609375e-06
v y w 2.384185791015625e-07
funcion 1.0013580322265625e-05
variables 1.430511474609375e-06
v y w 2.384185791015625e-07
funcion 9.775161743164062e-06



v 5.4836273193359375e-06
erfc 9.5367431640625e-07
v2 3.24249267578125e-05
definicion 7.152557373046875e-07
funcion 1.239776611328125e-05
v 4.5299530029296875e-06
erfc 7.152557373046875e-07
v2 2.8848648071289062e-05
definicion 4.76837158203125e-07
funcion 1.33514404296875e-05
v 5.245208740234375e-06
erfc 9.5367431640625e-07
v2 2.6941299438476562e-05
definicion 2.384185791015625e-07
funcion 1.0013580322265625e-05
v 5.0067901611328125e-06
erfc 1.1920928955078125e-06
v2 3.361701965332031e-05
definicion 4.76837158203125e-07
funcion 1.52587890625e-05

v 4.5299530029296875e-06
erfc 1.1920928955078125e-06
v2 3.8623809814453125e-05
definicion 9.5367431640625e-07
funcion 1.52587890625e-05
v 5.0067901611328125e-06
erfc 7.152557373046875e-07
v2 3.5762786865234375e-05
definicion 7.152557373046875e-07
funcion 1.1682510375976562e-05
v 5.0067901611328125e-06
erfc 9.5367431640625e-07
v2 3.647804260253906e-05
definicion 7.152557373046875e-07
funcion 1.1920928955078125e-05
v 4.76837158203125e-06
erfc 7.152557373046875e-07
v2 3.552436828613281e-05
definicion 7.152557373046875e-07
funcion 1.1682510375976562e-05