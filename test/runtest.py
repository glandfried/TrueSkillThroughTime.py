import unittest
import sys
sys.path.append('..')
import trueskillthroughtime as ttt
#import old
from importlib import reload  # Python 3.4+ only.
reload(ttt)
#reload(old)
import math

#import trueskill as ts
#env = ts.TrueSkill(draw_probability=0.0, beta=1.0, tau=0.0)
import time

class tests(unittest.TestCase):
    def test_gaussian_init(self):
        N01 = ttt.Gaussian(mu=0,sigma=1)
        self.assertAlmostEqual(N01.mu,0)
        self.assertAlmostEqual(N01.sigma,1.0)
        Ninf = ttt.Gaussian(0,math.inf)
        self.assertAlmostEqual(Ninf.mu,0)
        self.assertAlmostEqual(Ninf.sigma,math.inf)
        N00 = ttt.Gaussian(mu=0,sigma=0)
        self.assertAlmostEqual(N00.mu,0)
        self.assertAlmostEqual(N00.sigma,0)
    def test_ppf(self):
        self.assertAlmostEqual(ttt.ppf(0.3,ttt.N01.mu, ttt.N01.sigma),-0.52440044)
        N23 = ttt.Gaussian(2.,3.)
        self.assertAlmostEqual(ttt.ppf(0.3,N23.mu, N23.sigma),0.42679866)
    def test_cdf(self):
        self.assertAlmostEqual(ttt.cdf(0.3,ttt.N01.mu,ttt.N01.sigma),0.617911409)
        N23 = ttt.Gaussian(2.,3.)
        self.assertAlmostEqual(ttt.cdf(0.3,N23.mu,N23.sigma),0.28547031)
    def test_pdf(self):    
        self.assertAlmostEqual(ttt.pdf(0.3,ttt.N01.mu,ttt.N01.sigma),0.38138781)
        N23 = ttt.Gaussian(2.,3.)
        self.assertAlmostEqual(ttt.pdf(0.3,N23.mu,N23.sigma),0.11325579)
    def test_compute_margin(self):
        self.assertAlmostEqual(ttt.compute_margin(0.25,math.sqrt(2)*25.0/6),1.8776005988)
        self.assertAlmostEqual(ttt.compute_margin(0.25,math.sqrt(3)*25.0/6),2.29958170)
        self.assertAlmostEqual(ttt.compute_margin(0.0,math.sqrt(3)*25.0/6),2.7134875810435737e-07)
        self.assertAlmostEqual(ttt.compute_margin(1.0,math.sqrt(3)*25.0/6),math.inf)
    def test_trunc(self):
        mu, sigma = ttt.trunc(*ttt.Gaussian(0,1),0.,False)
        self.assertAlmostEqual((mu,sigma) ,(0.7978845368663289,0.6028103066716792) )
        mu, sigma = ttt.trunc(*ttt.Gaussian(0.,math.sqrt(2)*(25/6) ),1.8776005988,True)
        self.assertAlmostEqual(mu,0.0) 
        self.assertAlmostEqual(sigma,1.0767055, places=4)
        mu, sigma = ttt.trunc(*ttt.Gaussian(12.,math.sqrt(2)*(25/6)),1.8776005988,True)
        self.assertAlmostEqual(mu,0.3900995, places=5) 
        self.assertAlmostEqual(sigma,1.0343979, places=5)
    def gaussian(self):
        N, M = ttt.Gaussian(25.0, 25.0/3), ttt.Gaussian(0.0, 1.0)
        mu, sigma = M/N
        self.assertAlmostEqual(mu,-0.365, places=3) 
        self.assertAlmostEqual(sigma, 1.007, places=3) 
        mu, sigma = N*M
        self.assertAlmostEqual(mu,0.355, places=3) 
        self.assertAlmostEqual(sigma,0.993, places=3) 
        mu, sigma = N+M
        self.assertAlmostEqual(mu,25.000, places=3) 
        self.assertAlmostEqual(sigma,8.393, places=3) 
        mu, sigma = N - ttt.Gaussian(1.0, 1.0)
        self.assertAlmostEqual(mu,24.000, places=3) 
        self.assertAlmostEqual(sigma,8.393, places=3) 
    def test_dict_diff(self):
        self.assertEqual((0.1,0.05),ttt.max_tuple((0.,0.),(0.1,0.05)))
        self.assertEqual((0.1,0.05),ttt.max_tuple((0.,0.05),(0.1,0.0)))
        d1 = dict([(0, ttt.Gaussian(2.1,3.05))])
        d2 = dict([(0, ttt.Gaussian(2.0,3.0))])
        ttt.dict_diff(d1,d2)
    def test_1vs1(self):
        ta = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]
        g = ttt.Game([ta,tb],[0,1], 0.0)
        [a], [b] = g.posteriors()
        self.assertAlmostEqual(a.mu,20.79477925612302,4)
        self.assertAlmostEqual(b.mu,29.20522074387697,4)
        self.assertAlmostEqual(a.sigma,7.194481422570443 ,places=4)
        
        g = ttt.Game([[ttt.Player(ttt.Gaussian(29.,1.),25.0/6)] ,[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6)]], [0,1])
        [a], [b] = g.posteriors()
        self.assertAlmostEqual(a.mu,28.89648, places=4)
        self.assertAlmostEqual(a.sigma,0.9966043, places=4)
        self.assertAlmostEqual(b.mu,32.18921, places=4)
        self.assertAlmostEqual(b.sigma,6.062064, places=4)

        ta = [ttt.Player(ttt.Gaussian(1.139,0.531),1.0,0.2125)]
        tb = [ttt.Player(ttt.Gaussian(15.568,0.51),1.0,0.2125)]
        g = ttt.Game([ta,tb], [0,1], 0.0)
        self.assertAlmostEqual(g.likelihoods[0][0].sigma,ttt.inf)
        self.assertAlmostEqual(g.likelihoods[1][0].sigma,ttt.inf)
        self.assertAlmostEqual(g.likelihoods[0][0].mu,0.0)
    
    def test_1vs1vs1(self):
        [a], [b], [c] = ttt.Game([[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)],[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)],[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]], [1,2,0]).posteriors()
        self.assertAlmostEqual(a.mu,25.000000,5)
        self.assertAlmostEqual(a.sigma,6.238469796,5)
        self.assertAlmostEqual(b.mu,31.3113582213,5)
        self.assertAlmostEqual(b.sigma,6.69881865,5)
        self.assertAlmostEqual(c.mu,18.6886417787,5)
    
        [a], [b], [c] = ttt.Game([[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)],[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)],[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]]).posteriors()
        self.assertAlmostEqual(b.mu,25.000000,5)
        self.assertAlmostEqual(b.sigma,6.238469796,5)
        self.assertAlmostEqual(a.mu,31.3113582213,5)
        self.assertAlmostEqual(a.sigma,6.69881865,5)
        self.assertAlmostEqual(c.mu,18.6886417787,5)
    
        [a], [b], [c] = ttt.Game([[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)],[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)],[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]], [1,2,0],0.5).posteriors()
        self.assertAlmostEqual(a.mu,25.000,3)
        self.assertAlmostEqual(a.sigma,6.093,3)
        self.assertAlmostEqual(b.mu,33.379,3)
        self.assertAlmostEqual(b.sigma,6.484,3)
        self.assertAlmostEqual(c.mu,16.621,3)
        
    def test_1vs1_draw(self):
        [a], [b] = ttt.Game([[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)],[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]], [0,0], 0.25).posteriors()
        self.assertAlmostEqual(a.mu,25.000,2)
        self.assertAlmostEqual(a.sigma,6.469,2)
        self.assertAlmostEqual(b.mu,25.000,2)
        self.assertAlmostEqual(b.sigma,6.469,2)
        
        ta = [ttt.Player(ttt.Gaussian(25.,3.),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(29.,2.),25.0/6,25.0/300)]
        [a], [b] = ttt.Game([ta,tb], [0,0], 0.25).posteriors()
        self.assertAlmostEqual(a.mu,25.736,4)
        self.assertAlmostEqual(a.sigma,2.709956,4)
        self.assertAlmostEqual(b.mu,28.67289,4)
        self.assertAlmostEqual(b.sigma,1.916471,4)
    def draw_evidence_game(self):
        home = ttt.Player(ttt.Gaussian(0,0.001))
        away = ttt.Player(ttt.Gaussian(0,0.001))
        teams = [[home], [away]]
        result = [0, 0]
        g = ttt.Game(teams, result, p_draw=0.25)
        lhs = g.likelihoods[0][0]
        ev = g.evidence
        self.assertAlmostEqual(ev,0.25)
    def test_1vs1vs1_draw(self):
        [a], [b], [c] = ttt.Game([[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)],[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)],[ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,25.0/300)]], [0,0,0],0.25).posteriors()
        self.assertAlmostEqual(a.mu,25.000,3)
        self.assertAlmostEqual(a.sigma,5.729,3)
        self.assertAlmostEqual(b.mu,25.000,3)
        self.assertAlmostEqual(b.sigma,5.707,3)

        ta = [ttt.Player(ttt.Gaussian(25.,3.),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(25.,3.),25.0/6,25.0/300)]
        tc = [ttt.Player(ttt.Gaussian(29.,2.),25.0/6,25.0/300)]
        [a], [b], [c] = ttt.Game([ta,tb,tc], [0,0,0],0.25).posteriors()
        self.assertAlmostEqual(a.mu,25.489,3)
        self.assertAlmostEqual(a.sigma,2.638,3)
        self.assertAlmostEqual(b.mu,25.511,3)
        self.assertAlmostEqual(b.sigma,2.629,3)
        self.assertAlmostEqual(c.mu,28.556,3)
        self.assertAlmostEqual(c.sigma,1.886,3)
    def test_NvsN_Draw(self):
        ta = [ttt.Player(ttt.Gaussian(15.,1.),25.0/6,25.0/300),ttt.Player(ttt.Gaussian(15.,1.),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(30.,2.),25.0/6,25.0/300)]
        [a,b], [c] = ttt.Game([ta,tb], [0,0], 0.25).posteriors()
        self.assertAlmostEqual(a.mu,15.000,3)
        self.assertAlmostEqual(a.sigma,0.9916,3)
        self.assertAlmostEqual(b.mu,15.000,3)
        self.assertAlmostEqual(b.sigma,0.9916,3)
        self.assertAlmostEqual(c.mu,30.000,3)
        self.assertAlmostEqual(c.sigma,1.9320,3)
        
        [a,b], [c] = ttt.Game([ta,tb], [1,0], 0.0).posteriors()
        self.assertAlmostEqual(a.mu,15.105,3)
        self.assertAlmostEqual(a.sigma,0.995,3)
        
        ta = [ttt.Player(ttt.Gaussian(15.,1.),25.0/6,25.0/300),ttt.Player(ttt.Gaussian(15.,1.),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(15.,1.),25.0/6,25.0/300),ttt.Player(ttt.Gaussian(15.,1.),25.0/6,25.0/300)]
        [a,b], [c,d] = ttt.Game([ta,tb], [1,0], 0.0).posteriors()
        self.assertAlmostEqual(a.mu,15.093,3)
        self.assertAlmostEqual(a.sigma,0.996,3)
        self.assertAlmostEqual(c.mu,14.907,3)
        self.assertAlmostEqual(c.sigma,0.996,3)
        
    def test_NvsNvsN_mixt(self):
        ta = [ttt.Player(ttt.Gaussian(12.,3.),25.0/6,25.0/300)
             ,ttt.Player(ttt.Gaussian(18.,3.),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(30.,3.),25.0/6,25.0/300)]
        tc = [ttt.Player(ttt.Gaussian(14.,3.),25.0/6,25.0/300)
             ,ttt.Player(ttt.Gaussian(16.,3.),25.0/6,25.0/300)]
        [a,b], [c], [d,e]  = ttt.Game([ta,tb, tc], [1,0,0], 0.25).posteriors()
        self.assertAlmostEqual(a.mu,13.051,3)
        self.assertAlmostEqual(a.sigma,2.864,3)
        self.assertAlmostEqual(b.mu,19.051,3)
        self.assertAlmostEqual(b.sigma,2.864,3)
        self.assertAlmostEqual(c.mu,29.292,3)
        self.assertAlmostEqual(c.sigma,2.764,3)
        self.assertAlmostEqual(d.mu,13.658,3)
        self.assertAlmostEqual(d.sigma,2.813,3)
        self.assertAlmostEqual(e.mu,15.658,3)
        self.assertAlmostEqual(e.sigma,2.813,3)
    def test_evidence_1vs1(self):
        ta = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]
        g = ttt.Game([ta,tb], [0,0], 0.25)
        self.assertAlmostEqual(g.evidence,0.25,3)
        g = ttt.Game([ta,tb], [1,0], 0.25)
        self.assertAlmostEqual(g.evidence,0.375,3)
    def test_1vs1vs1_margin_0(self):
        ta = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]
        tc = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]

        proba = 0
        proba += ttt.Game([ta,tb,tc], [3,2,1], 0.).evidence
        proba += ttt.Game([ta,tb,tc], [3,1,2], 0.).evidence
        proba += ttt.Game([ta,tb,tc], [2,3,1], 0.).evidence
        proba += ttt.Game([ta,tb,tc], [1,3,2], 0.).evidence
        proba += ttt.Game([ta,tb,tc], [2,1,3], 0.).evidence
        proba += ttt.Game([ta,tb,tc], [1,2,3], 0.).evidence
   
        self.assertAlmostEqual(proba, 0.99527507)
    def test_forget(self):
        gamma = 0.15*25.0/3
        N = ttt.Gaussian(25.,1e-7)
        _, sigma = N.forget(gamma,5)
        self.assertAlmostEqual(sigma, math.sqrt(5*gamma**2))
        _, sigma = N.forget(gamma,1)
        self.assertAlmostEqual(sigma, math.sqrt(1*gamma**2))
    def test_one_event_each(self):
        agents = dict()
        for k in ["a", "b", "c", "d", "e", "f"]:
            agents[k] = ttt.Agent(ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300 ) , ttt.Ninf, -ttt.inf)
        b = ttt.Batch(composition=[ [["a"],["b"]], [["c"],["d"]] , [["e"],["f"]] ], results= [[1,0],[0,1],[1,0]], time = 0, agents=agents)
        post = b.posteriors()
        self.assertAlmostEqual(post["a"].mu,29.205,3)
        self.assertAlmostEqual(post["a"].sigma,7.194,3)
        
        self.assertAlmostEqual(post["b"].mu,20.795,3)
        self.assertAlmostEqual(post["b"].sigma,7.194,3)
        self.assertAlmostEqual(post["c"].mu,20.795,3)
        self.assertAlmostEqual(post["c"].sigma,7.194,3)
        self.assertEqual(b.convergence(),1)
    def test_batch_same_strength(self):
        agents = dict()
        for k in ["a", "b", "c", "d", "e", "f"]:
            agents[k] = ttt.Agent(ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300 ) , ttt.Ninf, -ttt.inf)
        b = ttt.Batch([ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ], [[1,0],[0,1],[1,0]], 2, agents)
        post = b.posteriors()
        self.assertAlmostEqual(post["a"].mu,24.96097,3)
        self.assertAlmostEqual(post["a"].sigma,6.299,3)
        self.assertAlmostEqual(post["b"].mu,27.09559,3)
        self.assertAlmostEqual(post["b"].sigma,6.01033,3)
        self.assertAlmostEqual(post["c"].mu,24.88968,3)
        self.assertAlmostEqual(post["c"].sigma,5.86631,3)
        self.assertEqual(b.convergence()>0, True)    
        post = b.posteriors()
        self.assertAlmostEqual(post["a"].mu,25.000,3)
        self.assertAlmostEqual(post["a"].sigma,5.419,3)
        self.assertAlmostEqual(post["b"].mu,25.000,3)
        self.assertAlmostEqual(post["b"].sigma,5.419,3)
        self.assertAlmostEqual(post["c"].mu,25.000,3)
        self.assertAlmostEqual(post["c"].sigma,5.419,3)
    def test_history_init(self):
        composition = [ [["aa"],["b"]], [["aa"],["c"]] , [["b"],["c"]] ]
        results = [[1,0],[0,1],[1,0]]
        priors = dict()
        for k in ["aa", "b", "c"]:
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 0.15*25.0/3) 
        
        h = ttt.History(composition, results, [1,2,3],priors)

        p0 = h.batches[0].posteriors()
        self.assertAlmostEqual(p0["aa"].mu,29.205,3)
        self.assertAlmostEqual(p0["aa"].sigma,7.19448,3)
        observed = h.batches[1].skills["aa"].forward.sigma 
        gamma = 0.15*25.0/3
        expected = math.sqrt((gamma*1)**2 +  h.batches[0].posterior("aa").sigma**2)
        self.assertAlmostEqual(observed, expected)
        observed = h.batches[1].posterior("aa")
        [expected], [c] = ttt.Game(h.batches[1].within_priors(0),[0,1]).posteriors()
        self.assertAlmostEqual(observed.mu, expected.mu, 3)
        self.assertAlmostEqual(observed.sigma, expected.sigma, 3)
    def test_one_batch_history(self):
        composition = [ [['aj'],['bj']],[['bj'],['cj']], [['cj'],['aj']] ]
        results = [[1,0],[1,0],[1,0]]
        times = [1,1,1]
        priors = dict()
        for k in ["aj", "bj", "cj"]:
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 0.15*25.0/3)
        h1 = ttt.History(composition,results, times,priors)
        self.assertAlmostEqual(h1.batches[0].posterior("aj").mu,22.904,3)
        self.assertAlmostEqual(h1.batches[0].posterior("aj").sigma,6.010,3)
        self.assertAlmostEqual(h1.batches[0].posterior("cj").mu,25.110,3)
        self.assertAlmostEqual(h1.batches[0].posterior("cj").sigma,5.866,3)
        step , i = h1.convergence()
        self.assertAlmostEqual(h1.batches[0].posterior("aj").mu,25.000,3)
        self.assertAlmostEqual(h1.batches[0].posterior("aj").sigma,5.419,3)
        self.assertAlmostEqual(h1.batches[0].posterior("cj").mu,25.000,3)
        self.assertAlmostEqual(h1.batches[0].posterior("cj").sigma,5.419,3)
    
        priors = dict()
        for k in ["aj", "bj", "cj"]:
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300)
        h2 = ttt.History(composition,results, [1,2,3], priors)
        self.assertAlmostEqual(h2.batches[2].posterior("aj").mu,22.904,3)
        self.assertAlmostEqual(h2.batches[2].posterior("aj").sigma,6.011,3)
        self.assertAlmostEqual(h2.batches[2].posterior("cj").mu,25.111,3)
        self.assertAlmostEqual(h2.batches[2].posterior("cj").sigma,5.867,3)
        step2 , i2 = h2.convergence()
        self.assertAlmostEqual(h2.batches[2].posterior("aj").mu,24.999,3)
        self.assertAlmostEqual(h2.batches[2].posterior("aj").sigma,5.420,3)
        self.assertAlmostEqual(h2.batches[2].posterior("cj").mu,25.001,3)
        self.assertAlmostEqual(h2.batches[2].posterior("cj").sigma,5.420,3)
    def test_trueSkill_Through_Time(self):
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1,0],[0,1],[1,0]]
        priors = dict()
        for k in ["a", "b", "c"]:
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300) 
        h = ttt.History(composition , results, [], priors)
        step , i = h.convergence()
        self.assertEqual(h.batches[2].skills["b"].elapsed, 1)
        self.assertEqual(h.batches[2].skills["c"].elapsed, 1)
        
        self.assertAlmostEqual(h.batches[0].posterior("a").mu,25.0002673,5)
        self.assertAlmostEqual(h.batches[0].posterior("a").sigma,5.41938162,5)
        self.assertAlmostEqual(h.batches[0].posterior("b").mu,24.999465,5)
        self.assertAlmostEqual(h.batches[0].posterior("b").sigma,5.419425831,5)
        self.assertAlmostEqual(h.batches[2].posterior("b").mu,25.00053219,5)
        self.assertAlmostEqual(h.batches[2].posterior("b").sigma,5.419696790,5)
    def test_env_TTT(self):
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1,0],[0,1],[1,0]]
        
        h = ttt.History(composition=composition, results=results, mu=25., sigma=25.0/3, beta=25.0/6, gamma=25.0/300)
        step , i = h.convergence()
        self.assertEqual(h.batches[2].skills["b"].elapsed, 1)
        self.assertEqual(h.batches[2].skills["c"].elapsed, 1)
        self.assertAlmostEqual(h.batches[0].posterior("a").mu,25.0002673,5)
        self.assertAlmostEqual(h.batches[0].posterior("a").sigma,5.41938162,5)
        self.assertAlmostEqual(h.batches[0].posterior("b").mu,24.999465,5)
        self.assertAlmostEqual(h.batches[0].posterior("b").sigma,5.419425831,5)
        self.assertAlmostEqual(h.batches[2].posterior("b").mu,25.00053219,5)
        self.assertAlmostEqual(h.batches[2].posterior("b").sigma,5.419696790,5)
    def test_env_0_TTT(self):
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1,0],[0,1],[1,0]]
        h = ttt.History(composition=composition, results=results, mu=0.0,sigma=6.0, beta=1.0, gamma=0.05)
        step , i = h.convergence(iterations=100)
        self.assertAlmostEqual(h.batches[0].posterior("a").mu,0.001,3)
        self.assertAlmostEqual(h.batches[0].posterior("a").sigma,2.395,3)
        self.assertAlmostEqual(h.batches[0].posterior("b").mu,-0.001,3)
        self.assertAlmostEqual(h.batches[0].posterior("b").sigma,2.396,3)
        self.assertAlmostEqual(h.batches[2].posterior("b").mu,0.001,3)
        self.assertAlmostEqual(h.batches[2].posterior("b").sigma,2.396,3)
        
        composition = [ [["a"],["b"]], [["c"],["a"]] , [["b"],["c"]] ]
        h = ttt.History(composition=composition, mu=0.0,sigma=6.0, beta=1.0, gamma=0.05)
        step , i = h.convergence(iterations=100)
        self.assertAlmostEqual(h.batches[0].posterior("a").mu,0.001,3)
        self.assertAlmostEqual(h.batches[0].posterior("a").sigma,2.395,3)
        self.assertAlmostEqual(h.batches[0].posterior("b").mu,-0.001,3)
        self.assertAlmostEqual(h.batches[0].posterior("b").sigma,2.396,3)
        self.assertAlmostEqual(h.batches[2].posterior("b").mu,0.001,3)
        self.assertAlmostEqual(h.batches[2].posterior("b").sigma,2.396,3)
        
    def test_teams(self):
        composition = [ [["a","b"],["c","d"]], [["e","f"] , ["b","c"]], [["a","d"], ["e","f"]]  ]
        results = [[1,0],[0,1],[1,0]]
        h = ttt.History(composition=composition, results=results, mu=0.0,sigma=6.0, beta=1.0, gamma=0.0)
        step, i = h.convergence()
        self.assertAlmostEqual(h.batches[0].posterior("a").mu,h.batches[0].posterior("b").mu,3)
        self.assertAlmostEqual(h.batches[0].posterior("a").sigma,h.batches[0].posterior("b").sigma,3)
        self.assertAlmostEqual(h.batches[0].posterior("c").mu,h.batches[0].posterior("d").mu,3)
        self.assertAlmostEqual(h.batches[0].posterior("c").sigma,h.batches[0].posterior("d").sigma,3)
        self.assertAlmostEqual(h.batches[1].posterior("e").mu,h.batches[1].posterior("f").mu,3)
        self.assertAlmostEqual(h.batches[1].posterior("e").sigma,h.batches[1].posterior("f").sigma,3)
        
        self.assertAlmostEqual(h.batches[0].posterior("a").mu,4.085,3)
        self.assertAlmostEqual(h.batches[0].posterior("a").sigma,5.107,3)
        self.assertAlmostEqual(h.batches[0].posterior("c").mu,-0.533,3)
        self.assertAlmostEqual(h.batches[0].posterior("c").sigma,5.107,3)
        self.assertAlmostEqual(h.batches[2].posterior("e").mu,-3.552,3)
        self.assertAlmostEqual(h.batches[2].posterior("e").sigma,5.155,3)
    def test_sigma_beta_0(self):
        composition = [ [["a","a_b","b"],["c","c_d","d"]]
                 , [["e","e_f","f"],["b","b_c","c"]]
                 , [["a","a_d","d"],["e","e_f","f"]]  ]
        results = [[1,0],[0,1],[1,0]]
        priors = dict()
        for k in ["a_b", "c_d", "e_f", "b_c", "a_d", "e_f"]:
            priors[k] = ttt.Player(ttt.Gaussian(mu=0.0, sigma=1e-7), beta=0.0, gamma=0.2)
        h = ttt.History(composition=composition, results=results, priors=priors, mu=0.0,sigma=6.0, beta=1.0, gamma=0.0)
        step , i = h.convergence()
        self.assertAlmostEqual(h.batches[0].posterior("a_b").mu,0.0,4)
        self.assertAlmostEqual(h.batches[0].posterior("a_b").sigma,0.0,4)
        self.assertAlmostEqual(h.batches[2].posterior("e_f").mu,-0.002,4)
        self.assertAlmostEqual(h.batches[2].posterior("e_f").sigma,0.2,4)
    def test_memory_Size(self):
        def summarysize(obj):
            import sys
            from types import ModuleType, FunctionType
            from gc import get_referents
            BLACKLIST = type, ModuleType, FunctionType
            if isinstance(obj, BLACKLIST):
                raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
            seen_ids = set()
            size = 0
            objects = [obj]
            while objects:
                need_referents = []
                for obj in objects:
                    if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                        seen_ids.add(id(obj))
                        size += sys.getsizeof(obj)
                        need_referents.append(obj)
                objects = get_referents(*need_referents)
            return size
    
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1,0],[0,1],[1,0]]
        h = ttt.History(composition =composition, results=results, times = [0, 10, 20], mu=0.0,sigma=6.0, beta=1.0, gamma=0.05)
        self.assertEqual( summarysize(h) < 13000 , True)
        self.assertEqual( summarysize(h.batches) - summarysize(h.agents) < 10000, True )
        self.assertEqual( summarysize(h.agents) < 3000, True )
        self.assertEqual( summarysize(h.batches[1]) - summarysize(h.agents) < 3500, True)
        self.assertEqual( summarysize(h) < summarysize(composition)*15, True)
        self.assertEqual( summarysize(h.batches[0].skills) < 1650, True)
        self.assertEqual( summarysize(h.batches[0].events) < 1850, True )
    def test_learning_curve(self):
        composition = [ [["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]] ]
        results = [[1,0],[1,0],[1,0]]    
        priors = dict()
        for k in ["aj", "bj", "cj"]:
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300)
        h = ttt.History(composition,results, [5,6,7], priors)
        h.convergence()
        lc = h.learning_curves()
        self.assertEqual(lc["aj"][0][0],5)
        self.assertEqual(lc["aj"][-1][0],7)
        self.assertAlmostEqual(lc["aj"][-1][1].mu,24.999,3)
        self.assertAlmostEqual(lc["aj"][-1][1].sigma,5.420,3)
        self.assertAlmostEqual(lc["cj"][-1][1].mu,25.001,3)
        self.assertAlmostEqual(lc["cj"][-1][1].sigma,5.420,3)
    def test_1vs1_with_weights(self):
        ta = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0)]
        wa = [1.0]
        tb = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0)]
        wb = [2.0]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox( ttt.Gaussian(30.625173, 7.765472), 1e-4))
        self.assertTrue(post[1][0].isapprox( ttt.Gaussian(13.749653, 5.733840), 1e-4))

        wa = [1.0]
        wb = [0.7]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox( ttt.Gaussian(27.630080, 7.206676), 1e-4))
        self.assertTrue(post[1][0].isapprox( ttt.Gaussian(23.158943, 7.801628), 1e-4))

        wa = [1.6]
        wb = [0.7]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox( ttt.Gaussian(26.142438, 7.573088), 1e-4))
        self.assertTrue(post[1][0].isapprox( ttt.Gaussian(24.500183, 8.193278), 1e-4))
        
        wa = [1.0]; wb = [0.0]
        ta = [ttt.Player(ttt.Gaussian(2.0,6.0),1.0,0.0)]
        tb = [ttt.Player(ttt.Gaussian(2.0,6.0),1.0,0.0)]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox( ttt.Gaussian(5.557176746, 4.0527906913), 1e-3))
        self.assertTrue(post[1][0].isapprox( ttt.Gaussian(2.0, 6.0), 1e-4))
        # NOTA: trueskill original tiene probelmas en la aproximación: post[1][0].mu = 1.999644 
        
        
        wa = [1.0]; wb = [-1.0]
        ta = [ttt.Player(ttt.Gaussian(2.0,6.0),1.0,0.0)]
        tb = [ttt.Player(ttt.Gaussian(2.0,6.0),1.0,0.0)]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox( post[1][0], 1e-4))
    def test_NvsN_with_weights(self):
        ta = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0), ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0)]
        wa = [0.4, 0.8]
        tb = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0), ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0)]
        wb = [0.9, 0.6]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox(  ttt.Gaussian(27.539023, 8.129639), 1e-4))
        self.assertTrue(post[0][1].isapprox(  ttt.Gaussian(30.078046, 7.485372), 1e-4))
        self.assertTrue(post[1][0].isapprox(  ttt.Gaussian(19.287197, 7.243465), 1e-4))
        self.assertTrue(post[1][1].isapprox(  ttt.Gaussian(21.191465, 7.867608), 1e-4))

        wa = [1.3, 1.5]
        wb = [0.7, 0.4]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox(  ttt.Gaussian(25.190190, 8.220511), 1e-4))
        self.assertTrue(post[0][1].isapprox(  ttt.Gaussian(25.219450, 8.182783), 1e-4))
        self.assertTrue(post[1][0].isapprox(  ttt.Gaussian(24.897589, 8.300779), 1e-4))
        self.assertTrue(post[1][1].isapprox(  ttt.Gaussian(24.941479, 8.322717), 1e-4))

        wa = [1.6, 0.2]
        wb = [0.7, 2.4]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox( ttt.Gaussian(31.674697, 7.501180), 1e-4))
        self.assertTrue(post[0][1].isapprox( ttt.Gaussian(25.834337, 8.320970), 1e-4))
        self.assertTrue(post[1][0].isapprox( ttt.Gaussian(22.079819, 8.180607), 1e-4))
        self.assertTrue(post[1][1].isapprox( ttt.Gaussian(14.987953, 6.308469), 1e-4))

    
        tc = [ttt.Player(ttt.Gaussian(25.0,25.0/3),25.0/6,0.0)]
        g = ttt.Game([ta,tc])
        post_2vs1 = g.posteriors()
        
        wa = [1.0, 1.0]
        wb = [1.0, 0.0]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox(post_2vs1[0][0], 1e-4))
        self.assertTrue(post[0][1].isapprox(post_2vs1[0][1], 1e-4))
        self.assertTrue(post[1][0].isapprox(post_2vs1[1][0], 1e-4))
        self.assertTrue(post[1][1].isapprox(tb[1].prior, 1e-4))
    def test_1vs1_TTT_with_weights(self):
        composition = [[["a"],["b"]], [["b"],["a"]]]
        weights = [[[5.0],[4.0]],[[5.0],[4.0]]]
        h = ttt.History(composition, mu=2.0, beta=1.0, sigma=6.0, gamma=0.0, weights=weights)
        lc = h.learning_curves()
        self.assertTrue(lc["a"][0][1].isapprox( ttt.Gaussian(5.53765944, 4.758722), 1e-4))
        self.assertTrue(lc["b"][0][1].isapprox( ttt.Gaussian(-0.83012755, 5.2395689), 1e-4))
        self.assertTrue(lc["a"][1][1].isapprox( ttt.Gaussian(1.7922776, 4.099566689), 1e-4))
        self.assertTrue(lc["b"][1][1].isapprox( ttt.Gaussian(4.8455331752, 3.7476161), 1e-4))
        
        h.convergence()
        lc = h.learning_curves()
        self.assertTrue(lc["a"][0][1].isapprox( ttt.Gaussian(lc["a"][0][1].mu, lc["a"][0][1].sigma), 1e-4))
        self.assertTrue(lc["b"][0][1].isapprox( ttt.Gaussian(lc["a"][0][1].mu, lc["a"][0][1].sigma), 1e-4))
        self.assertTrue(lc["a"][1][1].isapprox( ttt.Gaussian(lc["a"][0][1].mu, lc["a"][0][1].sigma), 1e-4))
        self.assertTrue(lc["b"][1][1].isapprox( ttt.Gaussian(lc["a"][0][1].mu, lc["a"][0][1].sigma), 1e-4))
        
        # In the julia tests but is this really doing anything?
        composition = [[["a"],["b"]], [["b"],["a"]]]
        weights = [[[1.0],[4.0]],[[5.0],[4.0]]]
        h = ttt.History(composition, mu=2.0, beta=1.0, sigma=6.0, gamma=0.0, weights=weights)
        lc = h.learning_curves()
    
    def gamma(self):
        composition = [ [["a"],["b"]], [["a"],["b"]]]
        results = [[1,0],[1,0]]
        
        h = ttt.History(composition=composition,results=results, mu=0.0,sigma=6.0, beta=1.0, gamma=0.0)
        mu0, sigma0 = h.batches[1].skills['a'].forward
        self.assertAlmostEqual(mu0, 3.33907896)
        self.assertAlmostEqual(sigma0, 4.98503276)
        
        h = ttt.History(composition=composition,results=results, mu=0.0,sigma=6.0, beta=1.0, gamma=10.0)
        mu10, sigma10 = h.batches[1].skills['a'].forward
        self.assertAlmostEqual(mu10, 3.33907896)
        self.assertAlmostEqual(sigma10, 11.1736543)
        
        #Observaci'on:
        #   El paquete trueskill python agrega gamma antes de la partida 
        #   devuelve, trueskill.Player(mu=6.555, sigma=9.645)
        
        h = ttt.History(composition=composition,results=results, mu=0.0,sigma=math.sqrt(6.0**2+10**2), beta=1.0, gamma=10.0)
        mu100, sigma100 = h.batches[0].posterior("a")
        self.assertAlmostEqual(mu100, 6.555467)
        self.assertAlmostEqual(sigma100, 9.6449906)

    ## add_history and add_events methods.

    def test_add_history(self):
        composition = [ [["a","b"],["c","d"]], [["e","f"] , ["b","c"]], [["a","d"], ["e","f"]]  ]
        results = [[1,0],[0,1],[1,0]]
        times =[100,300,500]
        h = ttt.History(composition=[composition[0]], results=[results[0]],times=[times[0]], mu=0.0,sigma=6.0, beta=1.0, gamma=0.0)
        h.convergence()
        h.add_history(composition=[composition[1]], results=[results[1]],times=[times[1]])
        h.add_history(composition=[composition[2]], results=[results[2]],times=[times[2]])
        h.learning_curves()

    def test_add_events_batch(self):
        pass
        #agents= dict()
        #for k in ["a", "b", "c", "d", "e", "f"]:
            #agents[k] = ttt.Agent(ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300 ) , ttt.Ninf, -ttt.inf)
        #composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        #results = [[1,0],[0,1],[1,0]]
        #b = ttt.Batch(composition = composition, results = results, time=0, agents = agents)
        #b.convergence()
        #b.add_events(composition,results)
        #self.assertEqual(len(b),6)
        #post = b.posteriors()
        #b.iteration(trace=True)
        #b.iteration(trace=True)
        #self.assertAlmostEqual(post["a"].mu,25.000,3)
        #self.assertAlmostEqual(post["a"].sigma,3.88,3)
        #self.assertAlmostEqual(post["b"].mu,25.000,3)
        #self.assertAlmostEqual(post["b"].sigma,3.88,3)
        #self.assertAlmostEqual(post["c"].mu,25.000,3)
        #self.assertAlmostEqual(post["c"].sigma,3.88,3)

    def ToDo(self):
        print("Ningun toDo")
        
if __name__ == "__main__":
    unittest.main()


