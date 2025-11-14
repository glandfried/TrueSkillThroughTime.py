import pandas as pd
# sudo pip3 install trueskillthroughtime
# import sys
# sys.path.append('..')
from trueskillthroughtime2 import *
import time
from datetime import datetime

# Data
df = pd.read_csv('input/history.csv', low_memory=False)

columns = zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id, df.double)
composition = [[[w1,w2],[l1,l2]] if d == 't' else [[w1],[l1]] for w1, w2, l1, l2, d in columns ]
times = [ datetime.strptime(t, "%Y-%m-%d").timestamp()/(60*60*24) for t in df.time_start]

start = time.time()
h = History(composition = composition, times = times, sigma = 1.6, gamma = 0.036)
h.convergence(epsilon=0.01, iterations=3)
end = time.time()
print(end-start)

lc = h.learning_curves()
lc["c0ej"] 
# OLD version [(17665.208333333332, N(mu=-0.668, sigma=1.341))]
# NEW version [(17665.208333333332, N(mu=-0.667463, sigma=1.341188))]
