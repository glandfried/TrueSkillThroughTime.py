import unittest
import sys
sys.path.append('..')
import src as ttt
import trueskill as ts
env = ts.TrueSkill(draw_probability=0.25)



class tests(unittest.TestCase):
    def test_trunc(self):
        a = ttt.Gaussian(0,1).trunc
        b = ttt.Gaussian(0.7978845368663289,0.6028103066716792)
        self.assertEqual(a, b)

    def test_1vs1(self):
        ta = [ttt.Rating()]
        tb = [ttt.Rating()]
        g = ttt.Game([ta,tb],[1,0])
        [a], [b] = g.posterior
        self.assertAlmostEqual(a.mu,20.79477925612302,4)
        self.assertAlmostEqual(b.mu,29.20522074387697,4)
        self.assertAlmostEqual(a.sigma,7.194481422570443 ,places=4)
        
    def test_1vs1vs1(self):
        [a], [b], [c] = ttt.Game([[ttt.Rating()],[ttt.Rating()],[ttt.Rating()]], [1,0,2]).posterior
        self.assertAlmostEqual(a.mu,25.000000,5)
        self.assertAlmostEqual(a.sigma,6.238469796,5)
        self.assertAlmostEqual(b.mu,31.3113582213,5)
        self.assertAlmostEqual(b.sigma,6.69881865,5)
        self.assertAlmostEqual(c.mu,18.6886417787,5)

env = ts.TrueSkill(draw_probability=0.25)
[a], [b] = env.rate([[env.Rating()],[env.Rating()]],[0,0])
a.sigma
if __name__ == "__main__":
    unittest.main()

"""
g = ttt.Game([[ttt.Rating()],[ttt.Rating()]],[1,0])

print(env.rate([[env.Rating()],[env.Rating()],[env.Rating()]],[1,0,2] ) )

print(g.m_t_ft)
print(g.compute_likelihood())
print(g.likelihood_analitico)

print(g.posterior)
"""
