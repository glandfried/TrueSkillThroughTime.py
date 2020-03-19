import src as th
#import trueskill as ts
import numpy as np
from importlib import reload  # Python 3.4+ only.
reload(th)
from collections import defaultdict
import matplotlib.pyplot as plt


time = th.Time([[[1],[2]],[[1],[3]],[[2],[3]]],[[0,1],[1,0],[0,1]])

time.forward_priors[1].posterior(th.Gaussian())
time.within_priors(1)

history = th.History([[1,2],[1,3],[[2],[3]]],[[0,1],[1,0],[0,1]])
history.forward_prior
trueskill_learning_curve = history.learning_curve.copy()

history.backpropagation()
history.backward_prior
history.propagation()
history.forward_prior
for k in history.learning_curve.keys():
    colores = ['red', 'blue', 'green']
    plt.plot(history.learning_curve[k],color=colores[k-1] )
    plt.plot(trueskill_learning_curve[k],color=colores[k-1])

learning_curve = defaultdict(lambda: [] ) 
forgeting_curve = defaultdict(lambda: [] ) 
for g in  history.games:
    for i in range(len(th.flat(g.names))):
        learning_curve[th.flat(g.names)[i]].append(th.flat(g.prior())[i])
        forgeting_curve[th.flat(g.names)[i]].append(th.flat(g.inverse_prior())[i])
for n in history.forward.keys():
    learning_curve[n].append(history.forward[n]) 
    forgeting_curve[n] = [history.backward[n]] + forgeting_curve[n]



history = th.History([[1,2],[1,3],[[2],[3]]]*5,[[0,1],[1,0],[0,1]]*5,default=th.Rating(mu=25,sigma=250,beta=25/6))
history.forward_prior
trueskill_learning_curve = history.learning_curve.copy()

history.backpropagation()
history.backward_prior
history.propagation()
history.forward_prior
for k in history.learning_curve.keys():
    plt.plot(history.learning_curve[k])

history = th.History([[1,2],[1,3],[[2],[3]]]*2+[[4,5],[4,6],[[5],[6]]]*2+[[3,6]]*4,[[0,1],[1,0],[0,1]]*4+[[0,1]]*4)
history.forward
history.backpropagation()
history.backward
history.propagation()
history.forward

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

###########
        
import kickscore as ks
from datetime import datetime
import time

observations = list()
clock = time.time() 

teams = [['1','2'],['1','3'],['2','3']]
results = [[0,1],[1,0],[0,1]]


for i in range(len(teams)):
    if int(results[i][0]) < int(results[i][1]):
        observations.append({ "winners": [teams[i][0]], "losers": [teams[i][1]], "t": clock +i})
    else:
        observations.append({ "winners": [teams[i][1]], "losers": [teams[i][0]], "t": clock +i})

seconds_in_year = 365.25 * 24 * 60 * 60

model = ks.BinaryModel()
kernel = (ks.kernel.Constant(var=0.03) + ks.kernel.Matern32(var=0.138, lscale=1.753*seconds_in_year))

for team in range(1,4):
    model.add_item(str(team), kernel=kernel)

for obs in observations:
    model.observe(**obs)

start_time = time.time()
converged = model.fit()
if converged:
    print("Model has converged.")
elapsed_time = time.time() - start_time
# 30 segundos, 3 partidas

model.item['1'].scores[1]
model.item['2'].scores[1]
model.item['3'].scores[1]

model.plot_scores(['1'], figsize=(14, 5));
model.plot_scores(['2'], figsize=(14, 5));
model.log_likelihood
import numpy as np
np.exp(model.log_likelihood/3)
