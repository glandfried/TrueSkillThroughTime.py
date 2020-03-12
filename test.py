import src as th
#import trueskill as ts
import numpy as np
from importlib import reload  # Python 3.4+ only.
reload(th)
from collections import defaultdict

history = th.History([[1,2],[1,3],[[2],[3]]],[[0,1],[1,0],[0,1]])
history.backpropagation()
history.forward
history.backward


#history.games[2].update
back_prior_history = defaultdict(lambda: [th.Skill(th.Gaussian())] ) 

for g in reversed(range(len(history))):#g=1
    
    inverse_prior = [ [ history.backward[n] for n in ns] for ns in history.games[g].names]
    for e in range(len(history.games[g])):#e=0
        history.games[g].teams[e].back_info(inverse_prior[e])
    likelihood = history.games[g].update
    inverse_posterior = [th.flat(inverse_prior)[i]*th.flat(likelihood)[i] for i in range(len(th.flat(history.games[g].names)))]
    history.update_backward(th.flat(history.games[g].names),th.flat(inverse_posterior) )
    
    flat_names = th.flat(history.games[g].names)
    for i in range(len(flat_names)):
        back_prior_history[flat_names[i]].append(history.backward[flat_names[i]])

for g in range(len(history)):
        
    
    
history.games[0].teams
history.games[1].teams
history.games[2].teams

# Prueba manual de TTT
# El caso más simple 

isinstance(s1_0, th.Gaussian)

s1_0 = th.Skill()
s2_0 = th.Skill()
s3_0 = th.Skill()

th.Gaussian(mu=s1_0,sigma= 3) 


s1_0.performance

teams = [[s1_0],[s2_0]]
teams_performances = [np.sum(list(map(lambda x: x.performance,te))) for te in teams]




te = th.Team([s1_0,s2_0])
te.t

g = th.Game([th.Team([s1_0]),th.Team([s2_0])], [0,1])
g.t

def backpropagation(self):
    delta = 0
    history.games[-1].last_likelihood
    return delta

        
    


