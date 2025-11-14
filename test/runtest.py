"""
Test Suite for TrueSkill Through Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive unit tests for the TrueSkill Through Time algorithm implementation.
Tests cover:
- Gaussian distribution operations
- Statistical functions (PDF, CDF, PPF)
- Single and multi-player games
- Team games
- Historical skill tracking
- Various game observation models (Ordinal, Continuous, Discrete)
- Edge cases and numerical stability

Run with: python runtest.py
"""

import unittest
import sys
sys.path.append('..')
import trueskillthroughtime2 as ttt
#import old
from importlib import reload  # Python 3.4+ only.
reload(ttt)
#reload(old)
import math
import numpy as np

#import trueskill as ts
#env = ts.TrueSkill(draw_probability=0.0, beta=1.0, tau=0.0)
import time


class tests(unittest.TestCase):
    """Test suite for TrueSkill Through Time algorithm."""

    def test_gaussian_init(self):
        """Test Gaussian distribution initialization with various parameters."""
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
        """Test percent point function (inverse CDF) for Gaussian distributions."""
        self.assertAlmostEqual(ttt.ppf(0.3,ttt.N01.mu, ttt.N01.sigma),-0.5244005127080409)
        N23 = ttt.Gaussian(2.,3.)
        self.assertAlmostEqual(ttt.ppf(0.3,N23.mu, N23.sigma),0.4267984618758771)

    def test_cdf(self):
        """Test cumulative distribution function for Gaussian distributions."""
        self.assertAlmostEqual(ttt.cdf(0.3,ttt.N01.mu,ttt.N01.sigma),0.617911409)
        N23 = ttt.Gaussian(2.,3.)
        self.assertAlmostEqual(ttt.cdf(0.3,N23.mu,N23.sigma),0.28547031)

    def test_pdf(self):
        """Test probability density function for Gaussian distributions."""    
        self.assertAlmostEqual(ttt.pdf(0.3,ttt.N01.mu,ttt.N01.sigma),0.38138781)
        N23 = ttt.Gaussian(2.,3.)
        self.assertAlmostEqual(ttt.pdf(0.3,N23.mu,N23.sigma),0.11325579)

    def test_compute_margin(self):
        """Test draw margin computation from draw probability."""
        self.assertAlmostEqual(ttt.compute_margin(0.25,math.sqrt(2)*25.0/6),1.8776004584348176)
        self.assertAlmostEqual(ttt.compute_margin(0.25,math.sqrt(3)*25.0/6),2.2995815319905395)
        self.assertAlmostEqual(ttt.compute_margin(0.0,math.sqrt(3)*25.0/6),0.0)
        self.assertAlmostEqual(ttt.compute_margin(1.0,math.sqrt(3)*25.0/6),math.inf)

    def test_trunc(self):
        """Test truncated Gaussian distribution parameters for win/draw outcomes."""
        mu, sigma = ttt.trunc(*ttt.Gaussian(0,1),0.,False)
        self.assertAlmostEqual(mu, 0.7978845368663289)
        self.assertAlmostEqual(sigma, 0.6028103066716792)
        mu, sigma = ttt.trunc(*ttt.Gaussian(0.,math.sqrt(2)*(25/6) ),1.8776005988,True)
        self.assertAlmostEqual(mu,0.0) 
        self.assertAlmostEqual(sigma,1.0767055, places=4)
        mu, sigma = ttt.trunc(*ttt.Gaussian(12.,math.sqrt(2)*(25/6)),1.8776005988,True)
        self.assertAlmostEqual(mu,0.3900995, places=5) 
        self.assertAlmostEqual(sigma,1.0343979, places=5)

    def gaussian(self):
        """Test Gaussian algebraic operations (add, subtract, multiply, divide)."""
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

    def test_1vs1(self):
        """Test 1v1 games with various skill distributions and parameters."""
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

    def test_1vs1vs1(self):
        """Test three-player free-for-all games with various outcomes."""
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
        """Test 1v1 games ending in a draw with various draw probabilities."""
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
        """Test evidence computation for draw games."""
        home = ttt.Player(ttt.Gaussian(0,0.001))
        away = ttt.Player(ttt.Gaussian(0,0.001))
        teams = [[home], [away]]
        result = [0, 0]
        g = ttt.Game(teams, result, p_draw=0.25)
        lhs = g.likelihoods[0][0]
        ev = g.evidence
        self.assertAlmostEqual(ev,0.25)

    def test_1vs1vs1_draw(self):
        """Test three-player games with draws and mixed outcomes."""
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
        """Test team vs team games with draws and various team sizes."""
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
        """Test complex multi-team games with mixed team sizes and outcomes."""
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
        """Test evidence (marginal likelihood) computation for 1v1 games."""
        ta = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]
        tb = [ttt.Player(ttt.Gaussian(25.,1e-7),25.0/6,25.0/300)]
        g = ttt.Game([ta,tb], [0,0], 0.25)
        self.assertAlmostEqual(g.evidence,0.25,3)
        g = ttt.Game([ta,tb], [1,0], 0.25)
        self.assertAlmostEqual(g.evidence,0.375,3)

    def test_1vs1vs1_margin_0(self):
        """Test that evidence sums to ~1 for all possible three-player outcomes."""
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

        self.assertAlmostEqual(proba, 0.9952751273757627)

    def test_12_factorial_combinations(self):
        """Test evidence for 12-player game approaches uniform distribution."""
        teams = []
        for i in range(12):
            teams.append([ttt.Player(ttt.Gaussian(0,0),1,0)])

        evidence = ttt.Game(teams, list(range(12)), 0.).evidence
        self.assertAlmostEqual(evidence, 1.2180818904254274e-08)
        self.assertAlmostEqual(1/479001600, 2.08767569878681e-09)
        self.assertAlmostEqual(evidence*479001600, 5.83463174444807)

    def test_history_init(self):
        """Test History initialization and forward propagation (TrueSkill mode)."""
        composition = [ [["aa"],["b"]], [["aa"],["c"]] , [["b"],["c"]] ]
        results = [[1,0],[0,1],[1,0]]
        priors = dict()
        for k in ["aa", "b", "c"]:
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 0.15*25.0/3)

        h = ttt.History(composition, results, [1,2,3],priors)
        h.forward_propagation() # TrueSkill

        p0 = h.learning_curves()
        self.assertAlmostEqual(p0["aa"][0][1].mu,29.205,3)
        self.assertAlmostEqual(p0["aa"][0][1].sigma,7.19448,3)

        observed = p0["aa"][1][1]

        [expected], [c] = ttt.Game(h.within_priors(b=1,e=0),[0,1]).posteriors()
        self.assertAlmostEqual(observed.mu, expected.mu, 3)
        self.assertAlmostEqual(observed.sigma, expected.sigma, 3)

    def test_one_batch_history(self):
        """Test single batch history with convergence iterations."""
        composition = [ [['aj'],['bj']],[['bj'],['cj']], [['cj'],['aj']] ]
        results = [[1,0],[1,0],[1,0]]
        times = [1,1,1]
        priors = dict()
        for k in ["aj", "bj", "cj"]:
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 0.15*25.0/3)
        h1 = ttt.History(composition,results, times,priors)
        h1.forward_propagation()
        self.assertAlmostEqual(h1.bskills[0]["aj"].posterior.mu,22.904,3)
        self.assertAlmostEqual(h1.bskills[0]["aj"].posterior.sigma,6.010,3)
        self.assertAlmostEqual(h1.bskills[0]["cj"].posterior.mu,25.110,3)
        self.assertAlmostEqual(h1.bskills[0]["cj"].posterior.sigma,5.866,3)
        step , i = h1.convergence(iterations=10,epsilon=0.0001, verbose=False)
        self.assertAlmostEqual(h1.bskills[0]["aj"].posterior.mu,25.000,3)
        self.assertAlmostEqual(h1.bskills[0]["aj"].posterior.sigma,5.419,3)
        self.assertAlmostEqual(h1.bskills[0]["cj"].posterior.mu,25.000,3)
        self.assertAlmostEqual(h1.bskills[0]["cj"].posterior.sigma,5.419,3)

        priors = dict()
        for k in ["aj", "bj", "cj"]:
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300)
        h2 = ttt.History(composition,results, [1,2,3], priors)
        h2.forward_propagation()
        self.assertAlmostEqual(h2.bskills[2]["aj"].posterior.mu,22.904,3)
        self.assertAlmostEqual(h2.bskills[2]["aj"].posterior.sigma,6.011,3)
        self.assertAlmostEqual(h2.bskills[2]["cj"].posterior.mu,25.111,3)
        self.assertAlmostEqual(h2.bskills[2]["cj"].posterior.sigma,5.867,3)
        #h1.backward_propagation()
        step , i = h2.convergence(iterations=10,epsilon=0.0001, verbose=False)
        self.assertAlmostEqual(h2.bskills[2]["aj"].posterior.mu,24.999,3)
        self.assertAlmostEqual(h2.bskills[2]["aj"].posterior.sigma,5.420,3)
        self.assertAlmostEqual(h2.bskills[2]["cj"].posterior.mu,25.001,3)
        self.assertAlmostEqual(h2.bskills[2]["cj"].posterior.sigma,5.420,3)

    def test_trueSkill_Through_Time(self):
        """Test TrueSkill Through Time algorithm with multiple batches."""
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1,0],[0,1],[1,0]]
        priors = {k: ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300) for k in ["a", "b", "c"]}
        h = ttt.History(composition , results, [], priors)
        # BIG CHANGE.
        #       Version 0 tenía elapsed 1 entre dos skills si no se le pasaban los tiempos
        #       Version 1 Si no se le pasa tiempo,
        #           Decidir:
        #               - todo ocurre en el mismo momento.
        #               - los tiempos son los indices de la posición (* la que se está usando)
        step , i = h.convergence(iterations=10,epsilon=0, verbose=False)
        #print(h.learning_curves())
        self.assertAlmostEqual(h.bskills[0]["a"].posterior.mu,25.000267,5)
        self.assertAlmostEqual(h.bskills[0]["a"].posterior.sigma,5.419423,5)
        self.assertAlmostEqual(h.bskills[0]["b"].posterior.mu,24.999198,5)
        self.assertAlmostEqual(h.bskills[0]["b"].posterior.sigma,5.419511,5)
        self.assertAlmostEqual(h.bskills[2]["b"].posterior.mu,25.001332,5)
        self.assertAlmostEqual(h.bskills[2]["b"].posterior.sigma,5.420053,5)

    def test_env_TTT(self):
        """Test TTT with custom environment parameters."""
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1,0],[0,1],[1,0]]

        h = ttt.History(composition=composition, results=results, mu=25., sigma=25.0/3, beta=25.0/6, gamma=25.0/300)
        step , i = h.convergence(verbose=False)
        #print(h.learning_curves())
        self.assertAlmostEqual(h.bskills[0]["a"].posterior.mu,25.000268,5)
        self.assertAlmostEqual(h.bskills[0]["a"].posterior.sigma,5.419423,5)
        self.assertAlmostEqual(h.bskills[0]["b"].posterior.mu,24.999198,5)
        self.assertAlmostEqual(h.bskills[0]["b"].posterior.sigma,5.419511,5)
        self.assertAlmostEqual(h.bskills[2]["b"].posterior.mu,25.001332,5)
        self.assertAlmostEqual(h.bskills[2]["b"].posterior.sigma,5.420053,5)

    def test_env_0_TTT(self):
        """Test TTT with zero-centered priors and custom parameters."""
        composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
        results = [[1,0],[0,1],[1,0]]
        h = ttt.History(composition=composition, results=results, mu=0.0,sigma=6.0, beta=1.0, gamma=0.05)
        step , i = h.convergence(iterations=14, verbose=False, epsilon=0)
        #print(h.learning_curves())
        self.assertAlmostEqual(h.bskills[0]["a"].posterior.mu,0.000548,5)
        self.assertAlmostEqual(h.bskills[0]["a"].posterior.sigma,2.395715,5)
        self.assertAlmostEqual(h.bskills[0]["b"].posterior.mu,-0.001641,5)
        self.assertAlmostEqual(h.bskills[0]["b"].posterior.sigma,2.395765,5)
        self.assertAlmostEqual(h.bskills[2]["b"].posterior.mu,0.001762,5)
        self.assertAlmostEqual(h.bskills[2]["b"].posterior.sigma,2.395930,5)

        composition = [ [["a"],["b"]], [["c"],["a"]] , [["b"],["c"]] ]
        h = ttt.History(composition=composition, mu=0.0,sigma=6.0, beta=1.0, gamma=0.05)
        step , i = h.convergence(iterations=14, epsilon=0, verbose=False)
        self.assertAlmostEqual(h.bskills[0]["a"].posterior.mu,0.000548,5)
        self.assertAlmostEqual(h.bskills[0]["a"].posterior.sigma,2.395715,5)
        self.assertAlmostEqual(h.bskills[0]["b"].posterior.mu,-0.001641,5)
        self.assertAlmostEqual(h.bskills[0]["b"].posterior.sigma,2.395765,5)
        self.assertAlmostEqual(h.bskills[2]["b"].posterior.mu,0.001762,5)
        self.assertAlmostEqual(h.bskills[2]["b"].posterior.sigma,2.395930,5)

    def test_teams(self):
        """Test team games through history tracking."""
        composition = [ [["a","b"],["c","d"]], [["e","f"] , ["b","c"]], [["a","d"], ["e","f"]]  ]
        results = [[1,0],[0,1],[1,0]]
        h = ttt.History(composition=composition, results=results, mu=0.0,sigma=6.0, beta=1.0, gamma=0.0)
        step, i = h.convergence(verbose=False)
        #print(h.learning_curves())
        self.assertAlmostEqual(h.bskills[0]["a"].posterior.mu,h.bskills[0]["b"].posterior.mu,5)
        self.assertAlmostEqual(h.bskills[0]["a"].posterior.sigma, h.bskills[0]["b"].posterior.sigma,5)
        self.assertAlmostEqual(h.bskills[0]["c"].posterior.mu,h.bskills[0]["d"].posterior.mu,3)
        self.assertAlmostEqual(h.bskills[0]["c"].posterior.sigma,h.bskills[0]["d"].posterior.sigma,3)
        self.assertAlmostEqual(h.bskills[1]["e"].posterior.mu,h.bskills[1]["f"].posterior.mu,3)
        self.assertAlmostEqual(h.bskills[1]["e"].posterior.sigma,h.bskills[1]["f"].posterior.sigma,3)

        self.assertAlmostEqual(h.bskills[0]["a"].posterior.mu,4.0849024,5)
        self.assertAlmostEqual(h.bskills[0]["a"].posterior.sigma,5.106919056,5)
        self.assertAlmostEqual(h.bskills[0]["c"].posterior.mu,-0.53302949,5)
        self.assertAlmostEqual(h.bskills[0]["c"].posterior.sigma,5.1069190,5)
        self.assertAlmostEqual(h.bskills[2]["e"].posterior.mu,-3.551872939,5)
        self.assertAlmostEqual(h.bskills[2]["e"].posterior.sigma,5.15456970,5)
    def test_sigma_beta_0(self):
        composition = [ [["a","a_b","b"],["c","c_d","d"]]
                 , [["e","e_f","f"],["b","b_c","c"]]
                 , [["a","a_d","d"],["e","e_f","f"]]  ]
        results = [[1,0],[0,1],[1,0]]
        priors = dict()
        for k in ["a_b", "c_d", "e_f", "b_c", "a_d", "e_f"]:
            priors[k] = ttt.Player(ttt.Gaussian(mu=0.0, sigma=1e-7), beta=0.0, gamma=0.2)
        h = ttt.History(composition=composition, results=results, priors=priors, mu=0.0,sigma=6.0, beta=1.0, gamma=0.0)
        step , i = h.convergence(verbose=False)
        self.assertAlmostEqual(h.bskills[0]["a_b"].posterior.mu,0.0,5)
        self.assertAlmostEqual(h.bskills[0]["a_b"].posterior.sigma,0.0,5)
        self.assertAlmostEqual(h.bskills[2]["e_f"].posterior.mu,-0.0019730,5)
        self.assertAlmostEqual(h.bskills[2]["e_f"].posterior.sigma,0.19998286,5)
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
        #print(summarysize(h))
        #print(summarysize(h.bskills))
        self.assertEqual( summarysize(h) < 9375 , True) # Antes 13000
        self.assertEqual( summarysize(h.bskills) < 5081 , True)
        self.assertEqual( summarysize(h.batches) < 1103 , True)
    def test_learning_curve(self):
        composition = [ [["aj"],["bj"]],[["bj"],["cj"]], [["cj"],["aj"]] ]
        results = [[1,0],[1,0],[1,0]]
        priors = dict()
        for k in ["aj", "bj", "cj"]:
            priors[k] = ttt.Player(ttt.Gaussian(25., 25.0/3), 25.0/6, 25.0/300)
        h = ttt.History(composition,results, [5,6,7], priors)
        h.convergence(verbose=False)
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
        self.assertTrue(post[0][0].isapprox( ttt.Gaussian(30.625173, 7.765472)))
        self.assertTrue(post[1][0].isapprox( ttt.Gaussian(13.749653, 5.733839)))

        wa = [1.0]
        wb = [0.7]
        g = ttt.Game([ta,tb], weights=[wa,wb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox( ttt.Gaussian(27.630081, 7.206677)))
        self.assertTrue(post[1][0].isapprox( ttt.Gaussian(23.158943, 7.801628)))

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
        h.forward_propagation()
        lc = h.learning_curves()
        self.assertTrue(lc["a"][0][1].isapprox( ttt.Gaussian(5.53765944, 4.758722), 1e-4))
        self.assertTrue(lc["b"][0][1].isapprox( ttt.Gaussian(-0.83012755, 5.2395689), 1e-4))
        self.assertTrue(lc["a"][1][1].isapprox( ttt.Gaussian(1.7922776, 4.099566689), 1e-4))
        self.assertTrue(lc["b"][1][1].isapprox( ttt.Gaussian(4.8455331752, 3.7476161), 1e-4))

        h.convergence(verbose=False, iterations=16, epsilon=0)
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
    def test_gamma(self):
        composition = [ [["a"],["b"]], [["a"],["b"]]]
        results = [[1,0],[1,0]]

        h = ttt.History(composition=composition,results=results, mu=0.0,sigma=6.0, beta=1.0, gamma=0.0)
        h.forward_propagation()
        mu0, sigma0 = h.bskills[1]['a'].forward
        self.assertAlmostEqual(mu0, 3.33907906)
        self.assertAlmostEqual(sigma0, 4.985032699)

        h = ttt.History(composition=composition,results=results, mu=0.0,sigma=6.0, beta=1.0, gamma=10.0)
        h.forward_propagation()
        mu10, sigma10 = h.bskills[1]['a'].forward
        self.assertAlmostEqual(mu10, 3.33907906260)
        self.assertAlmostEqual(sigma10, 11.1736543)

        #Observaci'on:
        #   El paquete trueskill python agrega gamma antes de la partida
        #   devuelve, trueskill.Player(mu=6.555, sigma=9.645)

        h = ttt.History(composition=composition,results=results, mu=0.0,sigma=math.sqrt(6.0**2+10**2), beta=1.0, gamma=10.0)
        h.forward_propagation()
        mu100, sigma100 = h.bskills[0]['a'].posterior
        self.assertAlmostEqual(mu100, 6.555467799)
        self.assertAlmostEqual(sigma100, 9.6449905098)
    def test_game_continuous_1vs1(self):
        ta = [ttt.Player(ttt.Gaussian(2,2),1,0)]
        tb = [ttt.Player(ttt.Gaussian(1,2),1,0)]
        #
        result_ta = 45.24
        result_tb = 44.24
        g = ttt.Game([ta,tb], [result_ta,result_tb], obs="Continuous")
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox(ttt.Gaussian(2,1.549193)))
        self.assertTrue(post[1][0].isapprox(ttt.Gaussian(1,1.549193)))

        g = ttt.Game([ta,tb], [result_ta,result_tb], obs="Continuous", weights=[[1.0],[1.0]])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox(ttt.Gaussian(2,1.549193)))
        self.assertTrue(post[1][0].isapprox(ttt.Gaussian(1,1.549193)))

        # Los pesos tienen un compartamiento raro, porque si la media es positiva un peso alto aumenta la habilidad, pero si la media es negativa, un peso bajo la disminuye. Esto no parece ser un comportamiento razonable teniendo en cuenta que el valor absoluto de las medias no tiene ningún significado. TODO: profundizar esta idea en el caso en el que el observable es "orden", pues esto mismo ocurre también ahí.
        g = ttt.Game([ta,tb], [result_ta,result_tb], obs="Continuous", weights=[[1.0],[2.0]])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox(ttt.Gaussian(2.160000,1.833030)))
        self.assertTrue(post[1][0].isapprox(ttt.Gaussian(0.680000,1.200000)))

        ta = [ttt.Player(ttt.Gaussian(0,2),1,0)]
        tb = [ttt.Player(ttt.Gaussian(-1,2),1,0)]
        #
        w_ta = [1.0]
        w_tb = [5.0]
        #
        g = ttt.Game([ta,tb], [result_ta,result_tb], obs="Continuous", weights=[w_ta,w_tb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox(ttt.Gaussian(-0.123077,1.968990), 1e-5))
        self.assertTrue(post[1][0].isapprox(ttt.Gaussian(-0.384615,0.960769), 1e-5))

        ta = [ttt.Player(ttt.Gaussian(2,2),1,0)]
        tb = [ttt.Player(ttt.Gaussian(1,2),1,0)]
        #
        w_ta = [1.0]
        w_tb = [5.0]
        #
        g = ttt.Game([ta,tb], [result_ta,result_tb], obs="Continuous", weights=[w_ta,w_tb])
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox(ttt.Gaussian(2.123077,1.968990) , 1e-5))
        self.assertTrue(post[1][0].isapprox(ttt.Gaussian(0.384615,0.960769), 1e-5))

    def test_game_continuous_NvsM(self):
        ta = [ttt.Player(ttt.Gaussian(2,2),1,0),
              ttt.Player(ttt.Gaussian(2,2),1,0)]
        tb = [ttt.Player(ttt.Gaussian(4,2),1,0)]
        tc = [ttt.Player(ttt.Gaussian(3,2),1,0)]
        result = [4.2, 0.2, 2.1]
        g = ttt.Game([ta,tc,tb], result, obs="Continuous")
        post = g.posteriors()
        self.assertTrue(post[0][0].isapprox(ttt.Gaussian(2.816000,1.649242) , 1e-5))
        self.assertTrue(post[0][1].isapprox(ttt.Gaussian(2.816000,1.649242) , 1e-5))
        self.assertTrue(post[1][0].isapprox(ttt.Gaussian(2.232000,1.442221) , 1e-5))
        self.assertTrue(post[2][0].isapprox(ttt.Gaussian(3.952000,1.442221) , 1e-5))

    def test_history_continuous_NvsMvsL(self):
        priors = dict()
        priors["a1"] = ttt.Player(ttt.Gaussian(2,2),1,0)
        priors["a2"] = ttt.Player(ttt.Gaussian(2,2),1,0)
        priors["b"] = ttt.Player(ttt.Gaussian(4,2),1,0)
        priors["c"] = ttt.Player(ttt.Gaussian(3,2),1,0)
        results = [[4.2, 0.2, 2.1]]
        obs = ["Continuous"]
        h = ttt.History(composition=[ [ ["a1", "a2"], ["c"], ["b"] ] ], results = results, priors=priors, obs=obs )
        h.forward_propagation()
        lc = h.learning_curves()
        self.assertTrue(lc["a1"][0][1].isapprox(ttt.Gaussian(2.816000,1.649242) , 1e-5))
        self.assertTrue(lc["a2"][0][1].isapprox(ttt.Gaussian(2.816000,1.649242) , 1e-5))
        self.assertTrue(lc["c"][0][1].isapprox(ttt.Gaussian(2.232000,1.442221) , 1e-5))
        self.assertTrue(lc["b"][0][1].isapprox(ttt.Gaussian(3.952000,1.442221) , 1e-5))


    def test_fixed_point_approx(self):
        r = 2 # score
        mu = 2 # mean difference
        sigma = 2 # sigma difference
        self.assertAlmostEqual(ttt.fixed_point_approx(r, mu, sigma),
                               (0.655192942490574, 0.6218258871945438))

    def test_game_discrete_1vs1(self):
        ta = [ttt.Player(ttt.Gaussian(4,6),1,0)]
        wa = [1.0]
        tb = [ttt.Player(ttt.Gaussian(0,6),1,0)]
        wb = [1.0]
        result = [0,54]
        g = ttt.Game([ta,tb], result = result, weights=[wa,wb], obs="Discrete")
        post= g.posteriors()
        #print(post)
        self.assertTrue(post[0][0].isapprox(ttt.Gaussian(0.118952,4.300102)))
        self.assertTrue(post[1][0].isapprox(ttt.Gaussian(3.881048,4.300102)))

        ta = [ttt.Player(ttt.Gaussian(4,1),1,0)]
        wa = [1.0]
        tb = [ttt.Player(ttt.Gaussian(0,1),1,0)]
        wb = [1.0]
        result = [math.exp(4),0]
        g = ttt.Game([ta,tb], result = result, weights=[wa,wb], obs="Discrete")
        post= g.posteriors()
        #print(post)
        self.assertTrue(post[0][0].isapprox(ttt.Gaussian(3.997732,0.866683)))
        self.assertTrue(post[1][0].isapprox(ttt.Gaussian(0.002268,0.866683)))

    def test_history_discrete_NvsMvsL(self):
        priors = dict()
        priors["a1"] = ttt.Player(ttt.Gaussian(2,2),1,0)
        priors["a2"] = ttt.Player(ttt.Gaussian(2,2),1,0)
        priors["b"] = ttt.Player(ttt.Gaussian(4,2),1,0)
        priors["c"] = ttt.Player(ttt.Gaussian(3,2),1,0)
        results = [[4, 0, 2]]
        obs = ["Discrete"]
        h = ttt.History(composition=[ [ ["a1", "a2"], ["c"], ["b"] ] ], results = results, priors=priors, obs=obs )
        h.forward_propagation()
        lc = h.learning_curves()
        #print(lc)
        self.assertTrue(lc["a1"][0][1].isapprox(ttt.Gaussian(2.059343,1.667489) , 1e-5))
        self.assertTrue(lc["a2"][0][1].isapprox(ttt.Gaussian(2.059343,1.667489) , 1e-5))
        self.assertTrue(lc["c"][0][1].isapprox(ttt.Gaussian(3.176773,1.482405) , 1e-5))
        self.assertTrue(lc["b"][0][1].isapprox(ttt.Gaussian(3.763883,1.463096) , 1e-5))

    def test_history_mixed_type_of_game_NvsMvsL(self):
        priors = dict()
        priors["a1"] = ttt.Player(ttt.Gaussian(2,2),1,0.1)
        priors["a2"] = ttt.Player(ttt.Gaussian(2,2),1,0.1)
        priors["b"] = ttt.Player(ttt.Gaussian(4,2),1,0.1)
        priors["c"] = ttt.Player(ttt.Gaussian(3,2),1,0.1)
        results = [[4,0,2],[4.0, 0.0, 2.1],[4, 0, 2]]
        times = [0, 1, 2]
        obs = ["Ordinal", "Continuous", "Discrete"]

        h = ttt.History(composition=[  [ ["a1", "a2"], ["c"], ["b"] ] ]*3, results = results, times=times, priors=priors, obs=obs )
        h.forward_propagation()
        lc = h.learning_curves()
        #print(lc)
        self.assertTrue(lc["a1"][0][1].isapprox(ttt.Gaussian(3.053757,1.782038) , 1e-5))
        self.assertTrue(lc["a1"][1][1].isapprox(ttt.Gaussian(2.996293,1.480994) , 1e-5))
        self.assertTrue(lc["a1"][2][1].isapprox(ttt.Gaussian(2.413075,1.269865) , 1e-5))

        self.assertTrue(lc["a2"][0][1].isapprox(ttt.Gaussian(3.053757,1.782038) , 1e-5))
        self.assertTrue(lc["c"][0][1].isapprox(ttt.Gaussian(1.925404,1.686565) , 1e-5))

    def test_add_history(self):
        primer_parte = [[['c'], ['e']], [['h'], ['e']], [['a'], ['f']], [['a'], ['b']], [['c'], ['f']], [['b'], ['e']], [['c'], ['b']], [['f'], ['e']], [['b'], ['h']], [['c'], ['e']], [['f'], ['h']], [['b'], ['h']], [['a'], ['e']], [['h'], ['d']], [['d'], ['h']], [['f'], ['a']]]

        h = ttt.History(composition=primer_parte, online = True)
        step, i = h.convergence(epsilon=0.0, iterations=16,verbose=False)

        h2 = ttt.History(composition=[], online=True)
        h2.add_history(primer_parte)
        step, i = h2.convergence(epsilon=0.0, iterations=16, verbose=False)

        lc1_online = h.learning_curves(who=["b"], online=True)
        lc1 = h.learning_curves(who=["b"])

        lc2_online = h2.learning_curves(who=["b"], online=True)
        lc2 = h2.learning_curves(who=["b"])
        for i in range(len(lc1["b"])):
            self.assertTrue(lc1["b"][i][1].isapprox(lc2["b"][i][1]))

    def test_pickle(self):
        import pickle

        primer_parte = [[['c'], ['e']], [['h'], ['e']], [['a'], ['f']], [['a'], ['b']], [['c'], ['f']], [['b'], ['e']], [['c'], ['b']], [['f'], ['e']], [['b'], ['h']], [['c'], ['e']], [['f'], ['h']], [['b'], ['h']], [['a'], ['e']], [['h'], ['d']], [['d'], ['h']], [['f'], ['a']]]

        h = ttt.History(composition=primer_parte, online=True)
        step, i = h.convergence(epsilon=0.0, iterations=16, verbose=False)

        with open('History.pickle', 'wb') as handle:
            pickle.dump(h, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('History.pickle', 'rb') as handle:
            h2 = pickle.load(handle)

        segunda_parte = [[['d'], ['b']], [['c'], ['f']], [['e'], ['a']], [['d'], ['g']], [['h'], ['c']], [['a'], ['g']], [['h'], ['c']], [['f'], ['d']], [['e'], ['d']], [['c'], ['b']], [['c'], ['g']], [['a'], ['d']], [['c'], ['a']], [['h'], ['b']], [['c'], ['b']], [['b'], ['c']]]

        h.add_history(composition=segunda_parte)
        step, i = h.convergence(epsilon=0.0, iterations=16, verbose=False)

        h2.add_history(composition=segunda_parte)
        step, i = h2.convergence(epsilon=0.0, iterations=16, verbose=False)

        lc2 = h2.learning_curves(online=True)
        lc = h.learning_curves(online=True)
        for i in range(len(lc2["c"])):
            self.assertTrue(lc2["c"][i][1].isapprox(lc["c"][i][1]))

    def test_history_log_evidence(self):

        primer_parte = [[['c'], ['e']], [['h'], ['e']], [['a'], ['f']], [['a'], ['b']], [['c'], ['f']], [['b'], ['e']], [['c'], ['b']], [['f'], ['e']], [['b'], ['h']], [['c'], ['e']], [['f'], ['h']], [['b'], ['h']], [['a'], ['e']], [['h'], ['d']], [['d'], ['h']], [['f'], ['a']]]

        h = ttt.History(composition=primer_parte)
        ho = ttt.History(composition=primer_parte, online=True)
        step, i = h.convergence(epsilon=0.0, iterations=16, verbose=False)
        step, i = ho.convergence(epsilon=0.0, iterations=16, verbose=False)

        self.assertAlmostEqual(ho.geometric_mean(), 0.528351402582294)
        self.assertAlmostEqual(h.geometric_mean(), 0.487397868228223)


if __name__ == "__main__":
    """Run all tests when script is executed directly."""
    unittest.main()


