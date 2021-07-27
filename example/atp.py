import pandas as pd
import sys
sys.path.append('..')
from trueskillthroughtime import *
import time
from datetime import datetime

# Data
df = pd.read_csv('input/history.csv', low_memory=False)

columns = zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id, df.double)
composition = [[[w1,w2],[l1,l2]] if d == 't' else [[w1],[l1]] for w1, w2, l1, l2, d in columns ]
times = [ datetime.strptime(t, "%Y-%m-%d").timestamp()/(60*60*24) for t in df.time_start]

#start = time.time()
h = History(composition = composition, times = times, sigma = 1.6, gamma = 0.036)
h.convergence(epsilon=0.01, iterations=10)
#end = time.time()
#print(end-start)
