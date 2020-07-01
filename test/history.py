import sys
import time
sys.path.append('../')
import src as ttt
from importlib import reload  # Python 3.4+ only.
reload(ttt)
tttEnv = ttt.TrueSkill()

history = tttEnv.History([[[1],[2,3]]],[[0,1]])
history.through_time()
history.times[0].posteriors

start = time.time()
history = tttEnv.History([[[1],[2,3]]]*100,[[0,1]]*100)
history.through_time(online=False)
history_elapsed = time.time() - start
len(history.times)

sum([t.time_analysis['likelihood'] for t in history.times])/history_elapsed
sum([t.time_analysis['posterior'] for t in history.times])/history_elapsed
sum([t.time_analysis['evidence'] for t in history.times])/history_elapsed
sum([t.time_analysis['game_init'] for t in history.times])/history_elapsed
sum([t.time_analysis['create_game'] for t in history.times])/history_elapsed
sum([t.time_analysis['iterations'] for t in history.times])
