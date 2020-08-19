# -*- coding: utf-8 -*-
"""
   Trueskill Through Time
   ~~~~~~~~~
   :copyright: (c) 2019-2020 by Gustavo Landfried.
   :license: BSD, see LICENSE for more details.
"""
import math
#import ipdb
#from numba import jit

"""
TODO:
    - NUMBA
"""

MU = 25.0; SIGMA = (MU/3)
PI = SIGMA**-2; TAU = PI * MU
BETA = (SIGMA / 2); GAMMA = (SIGMA / 100)
DRAW_PROBABILITY = 0.0
EPSILON = 1e-6
sqrt2 = math.sqrt(2)
sqrt2pi = math.sqrt(2 * math.pi)
inf = math.inf

def erfc(x):
    #"""(http://bit.ly/zOLqbc)"""
    z = abs(x)
    t = 1.0 / (1.0 + z / 2.0)
    a = -0.82215223 + t * 0.17087277; b =  1.48851587 + t * a
    c = -1.13520398 + t * b; d =  0.27886807 + t * c; e = -0.18628806 + t * d
    f =  0.09678418 + t * e; g =  0.37409196 + t * f; h =  1.00002368 + t * g
    r = t * math.exp(-z * z - 1.26551223 + t * h)
    return r if not(x<0) else 2.0 - r 

def erfcinv(y):
    if y >= 2: return -inf
    if y < 0: raise ValueError('argument must be nonnegative')
    if y == 0: return inf
    if not (y < 1): y = 2 - y
    t = math.sqrt(-2 * math.log(y / 2.0))
    x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t)
    for _ in [0,1,2]:
        err = erfc(x) - y
        x += err / (1.12837916709551257 * math.exp(-(x**2)) - x * err)
    return x if (y < 1) else -x

def compute_margin(draw_probability, size):
    _N = Gaussian(0.0, math.sqrt(size)*BETA)
    return abs(_N.ppf(0.5-draw_probability/2))



class Gaussian(object):
    def __init__(self,mu=0, sigma=inf, inverse=False):
        if not inverse:
            self.tau, self.pi = self.tau_pi(mu, sigma)
        else:
            self.tau, self.pi = mu, sigma
            
    def tau_pi(self,mu,sigma):
        if sigma < 0.: raise ValueError('sigma**2 should be greater than 0')
        if sigma > 0.:
            _pi = sigma**-2
            _tau = _pi * mu
        else:
            _pi = inf
            _tau = inf
        return _tau, _pi
    
    @property
    def mu(self):
        return 0. if (self.pi ==inf or self.pi==0.) else self.tau / self.pi
    @property
    def sigma(self):
        return math.sqrt(1 / self.pi) if self.pi else inf
    def cdf(self, x):
        z = -(x - self.mu) / (self.sigma * sqrt2)
        return (0.5 * erfc(z))
    def pdf(self, x):
        normalizer = (sqrt2pi * self.sigma)**-1
        functional = math.exp( -((x - self.mu)**2) / (2*self.sigma**2) ) 
        return normalizer * functional
    def ppf(self, p):
        return self.mu - self.sigma * sqrt2  * erfcinv(2 * p)
    def trunc(self, margin, tie):
        N01 = Gaussian(0,1)
        _alpha = (-margin-self.mu)/self.sigma
        _beta  = ( margin-self.mu)/self.sigma
        if not tie:
            #t= -_alpha
            v = N01.pdf(-_alpha) / N01.cdf(-_alpha)
            w = v * (v + (-_alpha))
        else:
            v = (N01.pdf(_alpha)-N01.pdf(_beta))/(N01.cdf(_beta)-N01.cdf(_alpha))
            u = (_alpha*N01.pdf(_alpha)-_beta*N01.pdf(_beta))/(N01.cdf(_beta)-N01.cdf(_alpha))
            w =  - ( u - v**2 ) 
        mu = self.mu + self.sigma * v
        sigma = self.sigma* math.sqrt(1-w)
        return Gaussian(mu, sigma)
    def __iter__(self):
        return iter((self.mu, self.sigma))
    def __repr__(self):
        return 'N(mu={:.3f}, sigma={:.3f})'.format(self.mu, self.sigma)
    def __add__(self, M):
        return Gaussian(self.mu + M.mu, math.sqrt(self.sigma**2 + M.sigma**2))
    def __sub__(self, M):
        return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 + M.sigma**2))
    def __mul__(self, M):
        _tau, _pi = self.tau + M.tau, self.pi + M.pi
        return Gaussian(_tau, _pi, inverse=True)        
    def __truediv__(self, M):
        _tau = self.tau - M.tau; _pi = self.pi - M.pi
        return Gaussian(_tau, _pi, inverse=True)        
    def delta(self, M):
        return abs(self.mu - M.mu) , abs(self.sigma - self.sigma) 
    def exclude(self, M):
        return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 - M.sigma**2) )
    def isapprox(self, M, tol=1e-4):
        return (abs(self.mu - M.mu) < tol) and (abs(self.sigma - M.sigma) < tol)
        
N01 = Gaussian(0,1)
N00 = Gaussian(0,0)
Nms = Gaussian(MU,SIGMA)
Ninf = Gaussian()

class Rating(object):
    def __init__(self, N=Nms, beta=BETA, gamma=GAMMA, name=""):
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.name = name
    
    def forget(self, t, max_sigma=SIGMA):
        _sigma = min(math.sqrt(self.N.sigma**2 + (self.gamma*t)**2), max_sigma)
        return Rating(Gaussian(self.N.mu, _sigma),self.beta,self.gamma,self.name)

    def performance(self):
        return Gaussian(self.N.mu, math.sqrt(self.N.sigma**2 + self.beta**2))
    
    def copy(self):
        return Rating(self.N, self.beta, self.gamma, self.name)

    def __repr__(self):
        return 'Rating(mu=%.3f, sigma=%.3f)' % (self.N.mu, self.N.sigma)

class team_messages(object):
    def __init__(self, prior=Ninf, likelihood_lose=Ninf, likelihood_win=Ninf):
        self.prior = prior
        self.likelihood_lose = likelihood_lose
        self.likelihood_win = likelihood_win
    @property
    def p(self):
        return self.prior*self.likelihood_lose*self.likelihood_win
    @property
    def posterior_win(self):
        return self.prior*self.likelihood_lose
    @property
    def posterior_lose(self):
        return self.prior*self.likelihood_win
    @property
    def likelihood(self):
        return self.likelihood_win*self.likelihood_lose

class diff_messages(object):
    def __init__(self, prior=Ninf, likelihood=Ninf):
        self.prior = prior
        self.likelihood = likelihood
    @property
    def p(self):
        return self.prior*self.likelihood

class Game(object):
    def __init__(self, teams, result, draw_proba=0.0):
        if len(teams) != len(result):
           raise ValueError("len(teams) != len(result)")
        if (0.0 > draw_proba) or (1.0 <= draw_proba):
           raise ValueError ("0.0 <= proba < 1.0")
        margin = draw_proba and compute_margin(draw_proba,sum([len(team) for team in teams]))
        self.teams = teams
        self.result = result
        self.margin = margin
        self.likelihoods = []
        self.evidence = 0.0
        self.compute_likelihoods()
        
    def __len__(self):
        return len(self.result)
    def size(self):
        return [len(team) for team in self.teams]
    def performance(self,i):
        res = N00
        for r in self.teams[i]:
            res += r.performance()
        return res
    def likelihood_teams(self):
        r = self.result
        o = sortperm(r) 
        t = [team_messages(self.performance(o[e]),Ninf, Ninf) for e in range(len(self))]
        d = [diff_messages(t[e].prior - t[e+1].prior, Ninf) for e in range(len(self)-1)]
        tie = [r[o[e]]==r[o[e+1]] for e in range(len(d))]
        self.evidence = 1.0
        for e in range(len(d)):
            self.evidence *= d[e].prior.cdf(self.margin)-d[e].prior.cdf(-self.margin) if tie[e] else 1-d[e].prior.cdf(self.margin)
        step = (inf, inf); i = 0 
        while gr_tuple(step,1e-6) and (i < 10):
            step = (0., 0.)
            for e in range(len(d)-1):
                d[e].prior = t[e].posterior_win - t[e+1].posterior_lose
                d[e].likelihood = d[e].prior.trunc(self.margin,tie[e])/d[e].prior
                likelihood_lose = t[e].posterior_win - d[e].likelihood
                step = max_tuple(step,t[e+1].likelihood_lose.delta(likelihood_lose))
                t[e+1].likelihood_lose = likelihood_lose
            for e in range(len(d)-1,0,-1):
                d[e].prior = t[e].posterior_win - t[e+1].posterior_lose
                d[e].likelihood = d[e].prior.trunc(self.margin,tie[e])/d[e].prior
                likelihood_win = t[e+1].posterior_lose + d[e].likelihood
                step = max_tuple(step,t[e].likelihood_win.delta(likelihood_win))
                t[e].likelihood_win = likelihood_win
            i += 1
        if len(d)==1:
            d[0].prior = t[0].posterior_win - t[1].posterior_lose
            d[0].likelihood = d[0].prior.trunc(self.margin,tie[0])/d[0].prior
        t[0].likelihood_win = t[1].posterior_lose + d[0].likelihood
        t[-1].likelihood_lose = t[-2].posterior_win - d[-1].likelihood
        return [ t[o[e]].likelihood for e in range(len(t))] 
    def compute_likelihoods(self):
        m_t_ft = self.likelihood_teams()
        self.likelihoods = [[ m_t_ft[e] - self.performance(e).exclude(self.teams[e][i].N) for i in range(len(self.teams[e])) ] for e in range(len(self))]
    @property
    def posteriors(self):
        return [[ self.likelihoods[e][i] * self.teams[e][i].N for i in range(len(self.teams[e]))] for e in range(len(self))]


class Batch(object):
    def __init__(self, events, results, time, last_time=dict() ,priors=dict()):
        if len(events)!= len(results): raise ValueError("len(events)!= len(results)")
        
        self.events = events
        self.results = results
        self.time = time
        self.elapsed = dict() 
        self.prior_forward = dict()
        self.prior_backward = dict()
        self.likelihoods = dict()
        self.old_within_prior = dict()
        self.evidences = [0 for _ in range(len(results))]
        self.partake = dict()
        self.agents = set( [a for event in events for teams in event for team in teams for a in team ] )
        for a in self.agents:
            self.partake[a] = [e for e in range(len(events)) for team in events[e] if a in team ]
            self.elapsed[a] = time - last_time[a] if a in last_time else 0
            self.prior_forward[a] = priors[a].forget(self.elapsed[a]) if a in priors else Rating(Nms,BETA,GAMMA,a) 
            self.prior_backward[a] = Ninf
            self.likelihoods[a] = dict()
            self.old_within_prior[a] = dict()
            for e in self.partake[a]:
                self.likelihoods[a][e] = Ninf
                self.old_within_prior[a][e] = self.prior_forward[a].N
            
        self.iteration()
        self.max_step = self.step_within_prior()
    
    def __repr__(self):
        return "Batch(time={}, events={}, results={})".format(self.time,self.events,self.results)
    def __len__(self):
        return len(self.results)
    def likelihood(self, agent):
        res = Ninf
        for k in self.likelihoods[agent]:
            res *= self.likelihoods[agent][k]
        return res
    def posterior(self, agent):
        return self.likelihood(agent)*self.prior_backward[agent]*self.prior_forward[agent].N   
    def posteriors(self):
        res = dict()
        for a in self.agents:
            res[a] = self.posterior(a)
        return res
    def within_prior(self, agent, k):
        res = self.prior_forward[agent].copy()
        res.N = self.posterior(agent)/self.likelihoods[agent][k]
        return res
    def within_priors(self, k):
        return [[self.within_prior(a, k) for a in team] for team in self.events[k]]
    def iteration(self):
        for e in range(len(self)):
            _priors = self.within_priors(e)
            teams = self.events[e]
                    
            for t in range(len(teams)):
                for j in range(len(teams[t])):
                    self.old_within_prior[teams[t][j]][e] = _priors[t][j].N
            
            g = Game(_priors, self.results[e])
            
            for t in range(len(teams)):
                for j in range(len(teams[t])):
                    self.likelihoods[teams[t][j]][e] = g.likelihoods[t][j] 
            
            self.evidences[e] = g.evidence
    def forward_prior_out(self, agent):
        res = self.prior_forward[agent].copy()
        res.N *= self.likelihood(agent)
        return res
    def backward_prior_out(self, agent):
        gamma = self.prior_forward[agent].gamma
        N = self.likelihood(agent)*self.prior_backward[agent]
        # IMPORTANTE: No usar el tope de forget ac\'a
        return N+Gaussian(0., gamma*self.elapsed[agent] ) 
    def step_within_prior(self):
        step = (0.,0.)
        for a in self.partake:
            if len(self.partake[a]) > 0:
                for e in self.partake[a]:
                    
                    step = max_tuple(step,self.old_within_prior[a][e].delta(self.within_prior(a, e).N) )
                    
        return step
    def convergence(self, epsilon=EPSILON):
        i = 0    
        while gr_tuple(self.max_step, epsilon) and (i < 10):
            self.iteration()
            self.max_step = self.step_within_prior()
            i += 1
        return i
    def new_backward_info(self, backward_message):
        for a in self.agents:
            self.prior_backward[a] = backward_message[a] if a in backward_message else Ninf
        self.max_step = (inf, inf)
        return self.convergence()
    
    def new_forward_info(self, forward_message):
        for a in self.agents:
            self.prior_forward[a] = forward_message[a].forget(self.elapsed[a]) if a in forward_message else Rating(Nms)
        self.max_step = (inf, inf)
        return self.convergence()
class History(object):
    def __init__(self,events,results,times=[],priors=dict()):
        if len(events) != len(results): raise ValueError("len(events) != len(results)")
        if (len(times) > 0) and (len(events) != len(times)): raise ValueError(" len(times) error ")
        self.size = len(events)
        self.times = times
        self.priors = priors
        self.forward_message = priors.copy()
        self.backward_message = dict()
        self.last_time = dict()
        self.batches = []
        self.agents = set( [a for event in events for teams in event for team in teams for a in team ] )
        self.trueskill(events,results)
        
    def __repr__(self):
        return "History(Size={}, Batches={}, Agents={})".format(self.size,len(self.batches),len(self.agents))
    def __len__(self):
        return self.size
    def trueskill(self, events, results):
        o = sortperm(self.times) if len(self.times)>0 else [i for i in range(len(events))]
        i = 0
        while i < len(self):
            j, t = i+1, 1 if len(self.times) == 0 else self.times[o[i]]
            while (len(self.times)>0) and (j < len(self)) and (self.times[o[j]] == t): j += 1
            b = Batch([events[k] for k in o[i:j]],[results[k] for k in o[i:j]], t, self.last_time, 
            self.forward_message)        
            self.batches.append(b)
            for a in b.agents:
                self.last_time[a] = t
                self.forward_message[a] = b.forward_prior_out(a)
            i = j
            
    def convergence(self,epsilon=EPSILON,iterations=10):
        step = (inf, inf); i = 0
        while gr_tuple(step, epsilon) and (i < iterations):
            step = (0., 0.)
            
            self.backward_message=dict()
            for j in reversed(range(len(self.batches)-1)):# j=2
                for a in self.batches[j+1].agents:# a = "c"
                    self.backward_message[a] = self.batches[j+1].backward_prior_out(a)
                old = self.batches[j].posteriors().copy()
                self.batches[j].new_backward_info(self.backward_message)
                step = max_tuple(step, dict_diff(old, self.batches[j].posteriors()))
            
            self.forward_message= self.priors.copy()
            for j in range(1,len(self.batches)):#j=2
                for a in self.batches[j-1].agents:#a = "b"
                    self.forward_message[a] = self.batches[j-1].forward_prior_out(a)
                old = self.batches[j].posteriors().copy()
                self.batches[j].new_forward_info(self.forward_message)
                step = max_tuple(step, dict_diff(old, self.batches[j].posteriors()))
        
            i += 1
        return step, i

def max_tuple(t1, t2):
    return max(t1[0],t2[0]), max(t1[1],t2[1])

def gr_tuple(tup, threshold):
    return (tup[0] > threshold) or (tup[1] > threshold)

def sortperm(xs):
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(xs))] 

def dict_diff(old, new):
    step = (0., 0.)
    for a in old:
        step = max_tuple(step, old[a].delta(new[a]))
    return step


                
