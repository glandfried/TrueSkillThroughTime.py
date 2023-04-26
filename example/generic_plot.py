from trueskillthroughtime import *
import random
import matplotlib.pyplot as plt

# Solve you own example
agents = ["a","b","c","d","e"]
composition = [ [[x] for x in random.choices(agents,k=2) ] for _ in range(1000)]
h = History(composition=composition, gamma=0.03, sigma=1.0)
h.convergence()

# Plot some learning curves
lc = h.learning_curves()
pp = plt.figure(); plt.xlabel("t"); plt.ylabel("skill")
cmap = plt.get_cmap("tab10")
for i, agent in enumerate(agents[0:3]):
    t = [v[0] for v in lc[agent]]
    mu = [v[1].mu for v in lc[agent]]
    sigma = [v[1].sigma for v in lc[agent]]
    plt.plot(t, mu, color= cmap(i), label=agent)
    plt.fill_between(t, [m+s for m,s in zip(mu, sigma)], [m-s for m,s in zip(mu, sigma)], alpha=0.2, color = cmap(i))

plt.legend()
plt.show()

