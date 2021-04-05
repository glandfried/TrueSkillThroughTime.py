import pandas as pd
import sys
sys.path.append('..')
from src import *
import time

# Data
df = pd.read_csv('input/history.csv', low_memory=False)

columns = zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id, df.double)
composition = [[[w1,w2],[l1,l2]] if d == 't' else [[w1],[l1]] for w1, w2, l1, l2, d in columns ]

tour_time = pd.to_datetime(df.time_start,format='%Y-%m-%d')
base_time = pd.to_datetime('1910-01-01',format='%Y-%m-%d')
times =  (tour_time-base_time).dt.days - df.round_number

start = time.time()
h = History(composition = composition, times = times, sigma = 1.6, gamma = 0.036)
h.convergence(epsilon=0.01, iterations=10)
end = time.time()
print(end-start)
