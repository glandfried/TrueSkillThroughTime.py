import sys
sys.path.append('..')
import trueskillthroughtime as ttt
from importlib import reload  # Python 3.4+ only.
reload(ttt)
from statistics import NormalDist
import random
AGENTS=1000
GAMES=100000
SIGMA_RANGE=10.0
MU_RANGE=10.0

def generate_agent_distribution():
    return [(i, random.gauss(mu=0.0, sigma=1.0)*MU_RANGE,random.gauss(mu=0.0, sigma=1.0)*SIGMA_RANGE) for i in range(AGENTS)]

def get_result(a, b):
    a_performance = random.gauss(mu=a[1], sigma=a[2])
    b_performance = random.gauss(mu=b[1], sigma=b[2])
    return [[a[0]],[b[0]]] if a_performance > b_performance else [[b[0]],[a[0]]]

def generate_composition(agents_distribution):
    composition=[]
    for i in range(GAMES):
        a,b = random.sample(agents_distribution, 2)
        composition.append(get_result(a,b))
    return composition

def setup_runtime():
    """Generate data to operate on"""
    agents_distribution = generate_agent_distribution()
    composition = generate_composition(agents_distribution)
    return composition

def test(composition):
       h = ttt.History(composition)
       h.convergence(verbose=False)

if __name__ == '__main__':
    import timeit
    print(timeit.timeit("test(composition)",number=1, setup="""
from __main__ import test, setup_runtime
composition = setup_runtime()
""")
)