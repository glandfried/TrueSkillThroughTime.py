import sys
sys.path.append('..')
from src import *
import timeit
import trueskill as ts

# Code 1
mu = 0.0; sigma = 6.0; beta = 1.0; gamma = 0.0
p1 = Player(Gaussian(mu, sigma), beta, gamma); p2 = Player(Gaussian(mu, sigma), beta, gamma)
p3 = Player(Gaussian(mu, sigma), beta, gamma); p4 = Player(Gaussian(mu, sigma), beta, gamma)

# Code 2
team_a = [ p1, p2 ]
team_b = [ p3, p4 ]
teams = [team_a, team_b]
g = Game(teams)
time_tt = timeit.timeit(lambda: Game(teams).posteriors, number=1000)/1000
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
time_tt = timeit.timeit(lambda: Game(teams, result, p_draw=0.25).posteriors, number=1000)/1000

# Code 6
c1 = [["a"],["b"]]
c2 = [["b"],["c"]]
c3 = [["c"],["a"]]
composition = [c1, c2, c3]
h = History(composition)
time_tt = timeit.timeit(lambda: History(composition), number=1000)/1000
h.convergence()
time_tt = timeit.timeit(lambda: h.convergence(iterations=1, verbose=False), number=10)/10

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

import time
start = time.time()
h = History(composition, results, times, priors, mu=2.0, gamma=0.015)
end = time.time()
end - start
start = time.time()
h.convergence(iterations=1)
end = time.time()
end - start

time_tt = timeit.timeit(lambda: History(composition, results, times, priors, mu=2.0, gamma=0.015), number=10)/10
time_tt = timeit.timeit(lambda: h.convergence(iterations=1), number=10)/10



env = ts.TrueSkill(mu = mu, sigma = sigma, beta = beta, tau= gamma, draw_probability=0.0)
r1 = env.Rating(); r2 = env.Rating(); r3 = env.Rating(); r4 = env.Rating()
time_tt = timeit.timeit(lambda: env.rate([[r1,r2],[r3,r4]]), number=10000)/10000

env = ts.TrueSkill(mu = mu, sigma = sigma, beta = beta, tau= gamma, draw_probability=0.25)
r1 = env.Rating(); r2 = env.Rating(); r3 = env.Rating(); r4 = env.Rating()
time_tt = timeit.timeit(lambda: env.rate([[r1],[r2,r3],[r4]],[0,1,1]), number=10000)/10000
 

