import pandas as pd
import numpy as np
import sys
sys.path.append('/home/mati/Storage/Tesis/AnalisisGo-Tesis/')
import TTT as thM
import time
from line_profiler import LineProfiler
import cProfile, pstats, io
from importlib import reload

reload(thM)
envM = thM.TrueSkill(draw_probability=0)
df = pd.read_csv('/home/mati/Storage/Tesis/AnalisisGo-Tesis/DatosPurificados/summary_filtered_handicapPositive.csv')
df=df[:5000]
df['date'] = df['started'].apply(lambda row: row[0:7])
#composition = [[[1],[2]], [[2],[3]], [[3],[1]]]
#results = [[0,1], [0,1], [0,1]]
#batch_number = [1,2,3]

from collections import defaultdict
#%%
prior_dict = defaultdict(lambda: envM.Rating(0, 25/3, 0, 1/100))
for h_key in set([(h, s) for h, s in zip(df.handicap, df.width)]):
    prior_dict[h_key]
results = list(df.black_win.map(lambda x: [1, 0] if x else [0, 1]))
composition = [[[w], [b]] if h < 2 else [[w], [b, (h, s)]] for w, b, h, s in zip(df.white, df.black, df.handicap, df.width)]
batch=list(df.date)

#startM = time.time()

pr = cProfile.Profile()
pr.enable()
historyM = envM.history(composition,results,batch_numbers=batch)

historyM.through_time(online=False)
    #profile = LineProfiler(historyM.convergence())
historyM.convergence()

#print('Mati',historyM.times[0].posteriors)
#endM = time.time()
pr.disable()
s = io.StringIO()
#pr.dump_stats('prof_data')
#ps = pstats.Stats('prof_data')
#ps.sort_stats(pstats.SortKey.CUMULATIVE)
#ps.dump_stats('TTTProfile')
#ps.print_stats()
#print('Mati', endM-startM)
#print(profile.print_stats())
ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
ps.print_stats()

with open('test-5k.txt', 'w+') as f:
    f.write(s.getvalue())
