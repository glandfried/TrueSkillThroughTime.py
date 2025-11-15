#sudo -H pip3 install trueskillthroughtime
from trueskillthroughtime import *
import timeit

print("Code 1")
mu = 0.0; sigma = 6.0; beta = 1.0; gamma = 0.0

print("Code 2")
p1 = Player(Gaussian(mu, sigma), beta, gamma); p2 = Player(Gaussian(mu, sigma), beta, gamma)
p3 = Player(Gaussian(mu, sigma), beta, gamma); p4 = Player(Gaussian(mu, sigma), beta, gamma)

print("Code 3")
team_a = [ p1, p2 ]
team_b = [ p3, p4 ]
teams = [team_a, team_b]
g = Game(teams)
time_tt = timeit.timeit(lambda: Game(teams).posteriors, number=1000)/1000
# g = Game(teams, [0,0])

print("Code 4")
lhs = g.likelihoods
ev = g.evidence
ev = round(ev, 3)
print(ev)

print("Code 5")
pos = g.posteriors()
print(pos[0][0])
print(lhs[0][0] * p1.prior)

print("Code 6")
ta = [p1]
tb = [p2, p3]
tc = [p4]
teams = [ta, tb, tc]
result = [1, 0, 0]
g = Game(teams, result, p_draw=0.25)
time_tt = timeit.timeit(lambda: Game(teams, result, p_draw=0.25).posteriors, number=1000)/1000

print("Code 7")
c1 = [["a"],["b"]]
c2 = [["b"],["c"]]
c3 = [["c"],["a"]]
composition = [c1, c2, c3]
h = History(composition, times=[0,0,0])
h.forward_propagation()
time_tt = timeit.timeit(lambda: History(composition), number=1000)/1000

print("Code 8")
lc = h.learning_curves()
print(lc["a"])
print(lc["b"])

print("Code 9")
h.convergence()
time_tt = timeit.timeit(lambda: h.convergence(iterations=1, verbose=False), number=10)/10
lc = h.learning_curves()
print(lc["a"])
print(lc["b"])

###
print("Code 10")
import math; from numpy.random import normal, seed; seed(99); N=1000
def skill(experience, middle, maximum, slope):
    return maximum/(1+math.exp(slope*(-experience+middle)))

target = [skill(i, 500, 2, 0.0075) for i in range(N)]
mus = []

print("Code 11")
for _ in range(1):
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

print("Skill evolution data: output/logistic.csv")
import pandas as pd
df = pd.DataFrame({"true" : target, "mu" : mu, "sigma" : sigma})
df.to_csv("output/logisitc.csv", index=False)

df = pd.DataFrame(mus)
df.to_csv("output/logisitcs_mu.csv", index=False)







