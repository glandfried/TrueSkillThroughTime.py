import sys
sys.path.append('..')
from src import *

# Code 1
mu = 0.0; sigma = 6.0; beta = 1.0; gamma = 0.0
p1 = Player(Gaussian(mu, sigma), beta, gamma); p2 = Player(Gaussian(mu, sigma), beta, gamma)
p3 = Player(Gaussian(mu, sigma), beta, gamma); p4 = Player(Gaussian(mu, sigma), beta, gamma)

# Code 2
team_a = [ p1, p2 ]
team_b = [ p3, p4 ]
teams = [team_a, team_b]
g = Game(teams)
# g = Game(teams, [0,0])

# Code 3
lhs = g.likelihoods
ev = g.evidence
ev = round(ev, 3)
print(ev)

# Code 4
pos = g.posteriors()
print(pos[0][0])
print(lhs[0][0] * p1.prior)

# Code 5
ta = [p1]
tb = [p2, p3]
tc = [p4]
teams = [ta, tb, tc]
result = [1, 0, 0]
g = Game(teams, result, p_draw=0.25)

# Code 6
c1 = [["a"],["b"]]
c2 = [["b"],["c"]]
c3 = [["c"],["a"]]
composition = [c1, c2, c3]
h = History(composition)
h.convergence()

# Code 7
lc = h.learning_curves()
print(lc["a"])
print(lc["b"])


###
import math; from numpy.random import normal, seed; seed(99); N=1000
def skill(experience, middle, maximum, slope):
    return maximum/(1+math.exp(slope*(-experience+middle)))

target = [skill(i, 500, 2, 0.0075) for i in range(N)]
mus = []
for _ in range(33):
    opponents = normal(target,0.5)
    composition = [[["a"], [str(i)]] for i in range(N)]
    results = [  [1.,0.] if normal(target[i]) > normal(opponents[i]) else [0.,1.] for i in range(N)]
    times = [i for i in range(N)]
    priors = dict([(str(i), Player(Gaussian(opponents[i], 0.2))) for i in range(N)])
    h = History(composition, results, times, priors, mu=2.0, gamma=0.015)
    h.convergence()
    mu = [tp[1].mu for tp in h.learning_curves()["a"]]
    sigma = [tp[1].sigma for tp in h.learning_curves()["a"]]
    mus.append(mu)

mus.append(target)

for m in mus:
    plt.plot(m, "--")

plt.plot(target, color="black")
plt.show()

#import matplotlib.pyplot as plt
#plt.plot(opponents)
#plt.show()
#plt.plot(mu_05)

import pandas as pd
df = pd.DataFrame({"true" : agent, "mu" : mu, "sigma" : sigma})
df.to_csv("output/logisitc.csv", index=False)


df = pd.DataFrame(mus)
df.to_csv("output/logisitcs_mu.csv", index=False)


#plt.plot(mu)
#plt.plot(agent)
#plt.show()






