import pandas as pd
import time
import kickscore as ks
from datetime import datetime
import time


df = pd.read_csv('input/history.csv', low_memory=False)

days = [ datetime.strptime(t, "%Y-%m-%d").timestamp()/(60*60*24) for t in df.time_start]

columns = zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id, df.double, days)
composition = [[[w1,w2],[l1,l2]] if d == 't' else [[w1],[l1]] for w1, w2, l1, l2, d, t in columns ]

agents = set( [a for teams in composition for team in teams for a in team] )

columns = zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id, df.double, days)
observations = [{"winners":[w1,w2],"losers":[l1,l2],"t":t} if d == 't' else {"winners":[w1],"losers":[l1],"t":t} for w1, w2, l1, l2, d, t in columns]

kernel = (ks.kernel.Constant(var=0.03))

start = time.time()
model = ks.BinaryModel()
for a in agents:
    model.add_item(a, kernel=kernel)


for obs in observations:
    model.observe(**obs)


converged = model.fit(verbose=True, max_iter=10)
end = time.time()
print(start - end)

