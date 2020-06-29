import time
import cProfile 
import pstats
import io
import numpy as np
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
    tsRes = ts.rate([[tsEnv.Rating()],[tsEnv.Rating()]],[0,1])
    tsTime += time.time() - start


tttTimes = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
for _ in range(100):    
    start = time.time()
    composition = [ttt.Team([ttt.Rating()]),ttt.Team([ttt.Rating()])]
    game = ttt.Game(composition,[0,1])
    end = time.time() - start
    tttTimes[0] += end

    tttTimes[1] += game.times_analysis['start'] #11%
    tttTimes[2] += game.times_analysis['diff'] # 11%
    tttTimes[3] += game.times_analysis['trunc'] #7,5%
    tttTimes[4] += game.times_analysis['end'] #24%
    tttTimes[5] += game.times_analysis['m_t_ft'] # sum(53%)
    tttTimes[6] += game.times_analysis['exclude'] #20%

tttTimes/tttTimes[0]
tttTimes[0]/100

from line_profiler import LineProfiler
lp = LineProfiler()
lp_wrapper = lp(ttt.Game(composition,[0,1]))
lp_wrapper(numbers)
lp.print_stats()

start = time.time()
isinstance(other, ttt.Gaussian)
(time.time() - start)*10


