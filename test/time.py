import time
import sys
sys.path.append('../')
import src as ttt
import trueskill as ts
from importlib import reload  # Python 3.4+ only.
reload(ttt)
tsEnv = ts.TrueSkill(draw_probability=0)

tsTime = 0
for _ in range(100):
    start = time.time()
    tsRes = tsEnv.rate([[tsEnv.Rating()],[tsEnv.Rating()]],[0,1])
    tsTime += time.time() - start

reload(ttt)
tttTimes = [0,0]
for _ in range(100):   
    composition = [ttt.Team([ttt.Rating()]),ttt.Team([ttt.Rating(),ttt.Rating()])] 
    start = time.time()
    game = ttt.Game(composition,[0,1])
    game.posterior
    end = time.time() - start
    tttTimes[0] += time.time() - start

tttEnv = ttt.TrueSkill()
history = tttEnv.History([[[1],[2,3]]],[[0,1]])
history.through_time()

tsTime/tttTimes[0]

    tttTimes[1] += game.time_likelihood

tttTimes[1]/tttTimes[0]

import math
a=list(range(100))
times_analysis = [0.0,0.0]
for _ in range(100):
    start = time.time()
    b = np.log10(a)
    times_analysis[0] += (time.time() - start)
    start = time.time()
    b = math.log10(a)
    times_analysis[1] += (time.time() - start)

