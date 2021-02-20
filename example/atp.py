import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from src import *

# Data
df = pd.read_csv('input/history.csv', low_memory=False)
composition = [[[w1,w2],[l1,l2]] if d == 't' else [[w1],[l1]] for w1, w2, l1, l2, d in zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id, df.double) ]   
times =  (pd.to_datetime(df.time_start,format='%Y-%m-%d')- pd.to_datetime('1910-01-01',format='%Y-%m-%d')).dt.days

history= History(composition = composition, times = times)
history.convergence()
