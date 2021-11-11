import pandas as pd
import sys
sys.path.append('..')
from trueskillthroughtime import *
import time
from datetime import datetime

# Data
df = pd.read_csv('input/history.csv', low_memory=False)

columns = zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id, df.double, df.ground)
composition = [[[w1,w1+g,w2,w2+g],[l1,l1+g,l2,l2+g]] if d == 't' else [[w1,w1+g],[l1,l1+g]] for w1, w2, l1, l2, d, g in columns ]
times = [ datetime.strptime(t, "%Y-%m-%d").timestamp()/(60*60*24) for t in df.time_start]

columns = zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id)
player_ids = set([ player for game in columns for player in game ])
priors = dict([(p, Player(Gaussian(0., 1.6), 1.0, 0.036) ) for p in player_ids])

h = History(composition = composition, times = times, beta = 0.0, sigma = 1.0, gamma = 0.01, priors = priors)
h.convergence(epsilon=0.01, iterations=10)
