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

# Code 7
lc = h.learning_curves()
print(lc["a"])
print(lc["b"])
