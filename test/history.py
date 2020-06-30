import sys
sys.path.append('../')
import src as ttt

tttEnv = ttt.TrueSkill()
history = tttEnv.History([[[1],[2,3]]],[[0,1]])
history.through_time()
history.times[0].posteriors