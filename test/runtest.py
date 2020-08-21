import unittest
import sys
sys.path.append('..')
import src as ttt
#import old
from importlib import reload  # Python 3.4+ only.
reload(ttt)
#reload(old)
import math

#import trueskill as ts
#env = ts.TrueSkill(draw_probability=0.25)
import time

#start = time.time()
#g = ttt.Game([[ttt.Rating(ttt.Gaussian(29,1))] ,[ttt.Rating()]], [1,0], 0.0)
#time.time() -start

#start = time.time()
#g = old.Game([[old.Rating(29,1)] ,[old.Rating()]], [1,0], 0.0)
#time.time() -start

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
        self.assertAlmostEqual(ttt.N01.ppf(0.3),-0.52440044)
        N23 = ttt.Gaussian(2.,3.)
        self.assertAlmostEqual(N23.ppf(0.3),0.42679866)
    def test_cdf(self):
        self.assertAlmostEqual(ttt.N01.cdf(0.3),0.617911409)
        N23 = ttt.Gaussian(2.,3.)
        self.assertAlmostEqual(N23.cdf(0.3),0.28547031)
    def test_pdf(self):    
        self.assertAlmostEqual(ttt.N01.pdf(0.3),0.38138781)
        N23 = ttt.Gaussian(2.,3.)
        self.assertAlmostEqual(N23.pdf(0.3),0.11325579)
    def test_compute_margin(self):
        self.assertAlmostEqual(ttt.compute_margin(0.25,2),1.8776005988)
        self.assertAlmostEqual(ttt.compute_margin(0.25,3),2.29958170)
        self.assertAlmostEqual(ttt.compute_margin(0.0,3),2.7134875810435737e-07)
        self.assertAlmostEqual(ttt.compute_margin(1.0,3),math.inf)
    def test_trunc(self):
        mu, sigma = ttt.Gaussian(0,1).trunc(0.,False)
        self.assertAlmostEqual((mu,sigma) ,(0.7978845368663289,0.6028103066716792) )
        mu, sigma = ttt.Gaussian(0.,math.sqrt(2)*(25/6) ).trunc(1.8776005988,True)
        self.assertAlmostEqual(mu,0.0) 
        self.assertAlmostEqual(sigma,1.07670, places=4)
        mu, sigma = ttt.Gaussian(12.,math.sqrt(2)*(25/6)).trunc(1.8776005988,True)
        self.assertAlmostEqual(mu,0.3900999, places=5) 
        self.assertAlmostEqual(sigma,1.034401, places=5)
    def test_1vs1(self):
        ta = [ttt.Rating()]
        tb = [ttt.Rating()]
        g = ttt.Game([ta,tb],[1,0])
        [a], [b] = g.posteriors
        self.assertAlmostEqual(a.mu,20.79477925612302,4)
        self.assertAlmostEqual(b.mu,29.20522074387697,4)
        self.assertAlmostEqual(a.sigma,7.194481422570443 ,places=4)
        
        g = ttt.Game([[ttt.Rating(ttt.Gaussian(29,1))] ,[ttt.Rating()]], [1,0], 0.0)
        [a], [b] = g.posteriors
        self.assertAlmostEqual(a.mu,28.896, places=2)
        self.assertAlmostEqual(a.sigma,0.996, places=2)
        self.assertAlmostEqual(b.mu,32.189, places=2)
        self.assertAlmostEqual(b.sigma,6.062, places=2)
    def test_1vs1vs1(self):
        [a], [b], [c] = ttt.Game([[ttt.Rating()],[ttt.Rating()],[ttt.Rating()]], [1,0,2]).posteriors
        self.assertAlmostEqual(a.mu,25.000000,5)
        self.assertAlmostEqual(a.sigma,6.238469796,5)
        self.assertAlmostEqual(b.mu,31.3113582213,5)
        self.assertAlmostEqual(b.sigma,6.69881865,5)
        self.assertAlmostEqual(c.mu,18.6886417787,5)
    
        [a], [b], [c] = ttt.Game([[ttt.Rating()],[ttt.Rating()],[ttt.Rating()]], [1,0,2],0.5).posteriors
        self.assertAlmostEqual(a.mu,25.000000,4)
        self.assertAlmostEqual(a.sigma,6.48760,4)
        self.assertAlmostEqual(b.mu,29.19950,4)
        self.assertAlmostEqual(b.sigma,7.00947,4)
        self.assertAlmostEqual(c.mu,20.80049,4)
    def test_1vs1_draw(self):
        [a], [b] = ttt.Game([[ttt.Rating()],[ttt.Rating()]], [0,0], 0.25).posteriors
        self.assertAlmostEqual(a.mu,25.000,2)
        self.assertAlmostEqual(a.sigma,6.469,2)
        self.assertAlmostEqual(b.mu,25.000,2)
        self.assertAlmostEqual(b.sigma,6.469,2)
        ta = [ttt.Rating(ttt.Gaussian(25.,3.))]
        tb = [ttt.Rating(ttt.Gaussian(29.,2.))]
        [a], [b] = ttt.Game([ta,tb], [0,0], 0.25).posteriors
        self.assertAlmostEqual(a.mu,25.736,2)
        self.assertAlmostEqual(a.sigma,2.710,2)
        self.assertAlmostEqual(b.mu,28.672,2)
        self.assertAlmostEqual(b.sigma,1.916,2)
    def test_1vs1vs1_draw(self):
        [a], [b], [c] = ttt.Game([[ttt.Rating()],[ttt.Rating()],[ttt.Rating()]], [0,0,0],0.25).posteriors
        self.assertAlmostEqual(a.mu,25.000,2)
        self.assertAlmostEqual(a.sigma,5.746947,4)
        self.assertAlmostEqual(b.mu,25.000,2)
        self.assertAlmostEqual(b.sigma,5.714755,4)

        ta = [ttt.Rating(ttt.Gaussian(25.,3.))]
        tb = [ttt.Rating(ttt.Gaussian(25.,3.))]
        tc = [ttt.Rating(ttt.Gaussian(29.,2.))]
        [a], [b], [c] = ttt.Game([ta,tb,tc], [0,0,0],0.25).posteriors
        self.assertAlmostEqual(a.mu,25.473,2)
        self.assertAlmostEqual(a.sigma,2.645,2)
        self.assertAlmostEqual(b.mu,25.505,2)
        self.assertAlmostEqual(b.sigma,2.631,2)
        self.assertAlmostEqual(c.mu,28.565,2)
        self.assertAlmostEqual(c.sigma,1.888,2)
    def test_NvsN_Draw(self):
        ta = [ttt.Rating(ttt.Gaussian(15.,1.)),ttt.Rating(ttt.Gaussian(15.,1.))]
        tb = [ttt.Rating(ttt.Gaussian(30.,2.))]
        [a,b], [c] = ttt.Game([ta,tb], [0,0], 0.25).posteriors
        self.assertAlmostEqual(a.mu,15.000,2)
        self.assertAlmostEqual(a.sigma,0.9916,3)
        self.assertAlmostEqual(b.mu,15.000,2)
        self.assertAlmostEqual(b.sigma,0.9916,3)
        self.assertAlmostEqual(c.mu,30.000,2)
        self.assertAlmostEqual(c.sigma,1.9320,2)
    def test_evidence_1vs1(self):
        ta = [ttt.Rating(ttt.Gaussian(25.,1e-7))]
        tb = [ttt.Rating(ttt.Gaussian(25.,1e-7))]
        g = ttt.Game([ta,tb], [0,0], 0.25)
        self.assertAlmostEqual(g.evidence,0.25,3)
        g = ttt.Game([ta,tb], [0,1], 0.25)
        self.assertAlmostEqual(g.evidence,0.375,3)

    def test_1vs1vs1_margin_0(self):
        ta = [ttt.Rating(ttt.Gaussian(25.,1e-7))]
        tb = [ttt.Rating(ttt.Gaussian(25.,1e-7))]
        tc = [ttt.Rating(ttt.Gaussian(25.,1e-7))]
        
        g_abc = ttt.Game([ta,tb,tc], [1,2,3], 0.)
        g_acb = ttt.Game([ta,tb,tc], [1,3,2], 0.)
        g_bac = ttt.Game([ta,tb,tc], [2,1,3], 0.)
        g_bca = ttt.Game([ta,tb,tc], [3,1,2], 0.)
        g_cab = ttt.Game([ta,tb,tc], [2,3,1], 0.)
        g_cba = ttt.Game([ta,tb,tc], [3,2,1], 0.)
        
        proba = 0
        proba += g_abc.evidence
        proba += g_acb.evidence
        proba += g_bac.evidence
        proba += g_bca.evidence
        proba += g_cab.evidence
        proba += g_cba.evidence            
        print("Corregir la evidencia multiequipos para que sume 1")
        self.assertAlmostEqual(proba, 1.49999991)
    def test_batch_one_event_each(self):
        b = ttt.Batch([ [["a"],["b"]], [["c"],["d"]] , [["e"],["f"]] ], [[0,1],[1,0],[0,1]], 2)
        post = b.posteriors()
        self.assertAlmostEqual(post["a"].mu,29.205,3)
        self.assertAlmostEqual(post["a"].sigma,7.194,3)
        
        self.assertAlmostEqual(post["b"].mu,20.795,3)
        self.assertAlmostEqual(post["b"].sigma,7.194,3)
        self.assertAlmostEqual(post["c"].mu,20.795,3)
        self.assertAlmostEqual(post["c"].sigma,7.194,3)
        self.assertEqual(b.convergence(),0)
    def test_batch_same_strength(self):
        b = ttt.Batch([ [["aa"],["b"]], [["aa"],["c"]] , [["b"],["c"]] ], [[0,1],[1,0],[0,1]], 2)
        post = b.posteriors()
        self.assertAlmostEqual(post["aa"].mu,24.96097,3)
        self.assertAlmostEqual(post["aa"].sigma,6.299,3)
        self.assertAlmostEqual(post["b"].mu,27.09559,3)
        self.assertAlmostEqual(post["b"].sigma,6.01033,3)
        self.assertAlmostEqual(post["c"].mu,24.88968,3)
        self.assertAlmostEqual(post["c"].sigma,5.86631,3)
        self.assertEqual(b.convergence()>0, True)    
        post = b.posteriors()
        self.assertAlmostEqual(post["aa"].mu,25.000,3)
        self.assertAlmostEqual(post["aa"].sigma,5.419,3)
        self.assertAlmostEqual(post["b"].mu,25.000,3)
        self.assertAlmostEqual(post["b"].sigma,5.419,3)
        self.assertAlmostEqual(post["c"].mu,25.000,3)
        self.assertAlmostEqual(post["c"].sigma,5.419,3)

    def test_history_init(self):
        events = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[0,1],[1,0],[0,1]]
        h = ttt.History(events, results, [1,2,3])

        self.assertEqual(not ttt.gr_tuple(h.batches[1].max_step, 1e-6) and  not ttt.gr_tuple(h.batches[1].max_step, 1e-6), True)
        p0 = h.batches[0].posteriors()
        self.assertAlmostEqual(p0["a"].mu,29.205,3)
        self.assertAlmostEqual(p0["a"].sigma,7.19448,3)
        observed = h.batches[1].prior_forward["a"].N.sigma 
        expected = math.sqrt((ttt.GAMMA*1)**2 +  h.batches[0].posterior("a").sigma**2)
        self.assertAlmostEqual(observed, expected)
        observed = h.batches[1].posterior("a")
        g = ttt.Game([[h.batches[1].prior_forward["a"]],[h.batches[1].prior_forward["c"]]],[1,0])
        [expected], [c] = g.posteriors
        self.assertAlmostEqual(observed.mu, expected.mu, 3)
        self.assertAlmostEqual(observed.sigma, expected.sigma, 3)
    def test_trueSkill_Through_Time(self):
        events = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[0,1],[1,0],[0,1]]
        h = ttt.History(events, results, [1,2,3])
        h.batches[0].posteriors()
        step , i = h.convergence()
        self.assertAlmostEqual(h.batches[0].posterior("a").mu,25.0002673,5)
        self.assertAlmostEqual(h.batches[0].posterior("a").sigma,5.41950697,5)
        self.assertAlmostEqual(h.batches[0].posterior("b").mu,24.9986633,5)
        self.assertAlmostEqual(h.batches[0].posterior("b").sigma,5.41968377,5)
        self.assertAlmostEqual(h.batches[2].posterior("b").mu,25.0029304,5)
        self.assertAlmostEqual(h.batches[2].posterior("b").sigma,5.42076739,5)
    def test_one_batch_history(self):
        composition = [ [['aj'],['bj']],[['bj'],['cj']], [['cj'],['aj']] ]
        results = [[0,1],[0,1],[0,1]]
        bache = [1,1,1]
        h1 = ttt.History(composition,results, bache)
        self.assertAlmostEqual(h1.batches[0].posterior("aj").mu,22.904,2)
        self.assertAlmostEqual(h1.batches[0].posterior("aj").sigma,6.010,2)
        self.assertAlmostEqual(h1.batches[0].posterior("cj").mu,25.110,2)
        self.assertAlmostEqual(h1.batches[0].posterior("cj").sigma,5.866,2)
        step , i = h1.convergence()
        self.assertAlmostEqual(h1.batches[0].posterior("aj").mu,25.000,2)
        self.assertAlmostEqual(h1.batches[0].posterior("aj").sigma,5.419,2)
        self.assertAlmostEqual(h1.batches[0].posterior("cj").mu,25.000,2)
        self.assertAlmostEqual(h1.batches[0].posterior("cj").sigma,5.419,2)
    
        h2 = ttt.History(composition,results, [1,2,3])
        self.assertAlmostEqual(h2.batches[2].posterior("aj").mu,22.904,2)
        self.assertAlmostEqual(h2.batches[2].posterior("aj").sigma,6.012,2)
        self.assertAlmostEqual(h2.batches[2].posterior("cj").mu,25.110,2)
        self.assertAlmostEqual(h2.batches[2].posterior("cj").sigma,5.867,2)
        step2 , i2 = h2.convergence()
        self.assertAlmostEqual(h2.batches[2].posterior("aj").mu,24.997,2)
        self.assertAlmostEqual(h2.batches[2].posterior("aj").sigma,5.421,2)
        self.assertAlmostEqual(h2.batches[2].posterior("cj").mu,25.000,2)
        self.assertAlmostEqual(h2.batches[2].posterior("cj").sigma,5.420,2)

if __name__ == "__main__":
    unittest.main()


