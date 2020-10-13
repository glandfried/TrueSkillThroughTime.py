# -*- coding: utf-8 -*-
"""
   Trueskill Through Time
   ~~~~~~~~~
   :copyright: (c) 2019-2020 by Gustavo Landfried.
   :license: BSD, see LICENSE for more details.
"""
import math
import timeit
from numba import njit, types, typed
#import ipdb
import trueskill as ts

"""
TODO:
    Optimize.
    - Numba have several problems with jitclass
    - c++ may be a better solution
"""

BETA = 1.0
MU = 0.0
SIGMA = BETA * 6
GAMMA = BETA * 0.05
P_DRAW = 0.0
EPSILON = 1e-6
ITERATIONS = 10
sqrt2 = math.sqrt(2)
sqrt2pi = math.sqrt(2 * math.pi)
inf = math.inf
PI = SIGMA**-2
TAU = PI * MU

class Environment(object):
    def __init__(self, mu=MU, sigma=SIGMA, beta=BETA, gamma=GAMMA, p_draw=P_DRAW, epsilon=EPSILON, iterations=ITERATIONS):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.gamma = gamma
        self.p_draw = p_draw
        self.epsilon = epsilon
        self.iterations = iterations

@njit(types.f8(types.f8))
def erfc(x):
    #"""(http://bit.ly/zOLqbc)"""
    z = abs(x)
    t = 1.0 / (1.0 + z / 2.0)
    a = -0.82215223 + t * 0.17087277; b =  1.48851587 + t * a
    c = -1.13520398 + t * b; d =  0.27886807 + t * c; e = -0.18628806 + t * d
    f =  0.09678418 + t * e; g =  0.37409196 + t * f; h =  1.00002368 + t * g
    r = t * math.exp(-z * z - 1.26551223 + t * h)
    return r if not(x<0) else 2.0 - r 

#timeit.timeit(lambda: erfc(0.9) , number=10000)/10000

@njit(types.f8(types.f8))
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

#timeit.timeit(lambda: erfcinv(0.9) , number=10000)/10000

@njit(types.UniTuple(types.f8, 2)(types.f8,types.f8))
def tau_pi(mu,sigma):
    if sigma > 0.0:
        pi_ = sigma ** -2
        tau_ = pi_ * mu
    elif (sigma + 1e-5) < 0.0:
        raise ValueError(" sigma should be greater than 0 ")
    else:
        pi_ = inf
        tau_ = inf
    return tau_, pi_

@njit(types.UniTuple(types.f8, 2)(types.f8,types.f8))
def mu_sigma(tau_,pi_):
    if pi_ > 0.0:
        sigma = math.sqrt(1/pi_)
        mu = tau_ / pi_
    elif pi_ + 1e-5 < 0.0:
        raise ValueError(" sigma should be greater than 0 ")
    else:
        sigma = inf 
        mu = 0.0
    return mu, sigma
        
#timeit.timeit(lambda: mu_sigma(1.0,2.0) , number=10000)/10000

@njit(types.f8(types.f8,types.f8,types.f8))
def cdf(x, mu, sigma):
    z = -(x - mu) / (sigma * sqrt2)
    return (0.5 * erfc(z))

@njit(types.f8(types.f8,types.f8,types.f8))
def pdf(x, mu, sigma):
    normalizer = (sqrt2pi * sigma)**-1
    functional = math.exp( -((x - mu)**2) / (2*sigma**2) ) 
    return normalizer * functional

@njit(types.f8(types.f8,types.f8,types.f8))
def ppf(p, mu, sigma):
    return mu - sigma * sqrt2  * erfcinv(2 * p)

@njit(types.UniTuple(types.f8, 2)(types.f8,types.f8,types.f8,types.f8))
def trunc(mu, sigma, margin, tie):
    if not tie:
        _alpha = (margin-mu)/sigma
        v = pdf(-_alpha,0,1) / cdf(-_alpha,0,1)
        w = v * (v + (-_alpha))
    else:
        _alpha = (-margin-mu)/sigma
        _beta  = ( margin-mu)/sigma
        v = (pdf(_alpha,0,1)-pdf(_beta,0,1))/(cdf(_beta,0,1)-cdf(_alpha,0,1))
        u = (_alpha*pdf(_alpha,0,1)-_beta*pdf(_beta,0,1))/(cdf(_beta,0,1)-cdf(_alpha,0,1))
        w =  - ( u - v**2 ) 
    mu_trunc = mu + sigma * v
    sigma_trunc = sigma * math.sqrt(1-w)
    return mu_trunc, sigma_trunc

#timeit.timeit(lambda: trunc(1.0,2.0,0.0,False), number=10000)/10000



@njit(types.f8(types.f8,types.f8))
def compute_margin(p_draw, sd):
    return abs(ppf(0.5-p_draw/2, 0.0, sd ))

def max_tuple(t1, t2):
    return max(t1[0],t2[0]), max(t1[1],t2[1])

def gr_tuple(tup, threshold):
    return (tup[0] > threshold) or (tup[1] > threshold)

def sortperm(xs):
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(xs))] 


class Gaussian(object):
    def __init__(self,mu=MU, sigma=SIGMA):
        if sigma >= 0.0:
            self.mu, self.sigma = mu, sigma
        else:
            raise ValueError(" sigma should be greater than 0 ")
    
    @property
    def tau(self):
        if self.sigma > 0.0:
            return self.mu * (self.sigma**-2)
        else:
            return inf
        
    @property
    def pi(self):
        if self.sigma > 0.0:
            return self.sigma**-2
        else:
            return inf
    
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
        mu, sigma = mu_sigma(_tau, _pi)
        return Gaussian(mu, sigma)        
    def __truediv__(self, M):
        _tau = self.tau - M.tau; _pi = self.pi - M.pi
        mu, sigma = mu_sigma(_tau, _pi)
        return Gaussian(mu, sigma)
    def forget(self,gamma,t):
        return Gaussian(self.mu, math.sqrt(self.sigma**2 + t*gamma**2))
    def delta(self, M):
        return abs(self.mu - M.mu) , abs(self.sigma - self.sigma) 
    def exclude(self, M):
        return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 - M.sigma**2) )
    def isapprox(self, M, tol=1e-4):
        return (abs(self.mu - M.mu) < tol) and (abs(self.sigma - M.sigma) < tol)
    
#timeit.timeit(lambda: Gaussian(1.0,2.0) , number=10000)/10000

N01 = Gaussian(0,1)
N00 = Gaussian(0,0)
Ninf = Gaussian(0,inf)
Nms = Gaussian(MU, SIGMA)

class Rating(object):
    def __init__(self, mu=MU, sigma=SIGMA, beta=BETA, gamma=GAMMA, draw=Ninf):
        self.N = Gaussian(mu,sigma)
        self.beta = beta
        self.gamma = gamma
        self.draw = draw
 
    def performance(self):
        return Gaussian(self.N.mu, math.sqrt(self.N.sigma**2 + self.beta**2))
    def __repr__(self):
        return 'Rating(mu=%.3f, sigma=%.3f)' % (self.N.mu, self.N.sigma) 
 
class team_messages(object):
    def __init__(self, prior=Ninf, likelihood_lose=Ninf, likelihood_win=Ninf, likelihood_draw=Ninf):
        self.prior = prior
        self.likelihood_lose = likelihood_lose
        self.likelihood_win = likelihood_win
        self.likelihood_draw = likelihood_draw
        
    @property
    def p(self):
        return self.prior*self.likelihood_lose*self.likelihood_win*self.likelihood_draw
    @property
    def posterior_win(self):
        return self.prior*self.likelihood_lose*self.likelihood_draw
    @property
    def posterior_lose(self):
        return self.prior*self.likelihood_win*self.likelihood_draw
    @property
    def likelihood(self):
        return self.likelihood_win*self.likelihood_lose*self.likelihood_draw

class draw_messages(object):
    def __init__(self,prior = Ninf, prior_team = Ninf, likelihood_lose = Ninf, likelihood_win = Ninf):
        self.prior = prior
        self.prior_team = prior_team
        self.likelihood_lose = likelihood_lose
        self.likelihood_win = likelihood_win
    
    @property
    def p(self):
        return self.prior_team*self.likelihood_lose*self.likelihood_win
    
    @property
    def posterior_win(self):
        return self.prior_team*self.likelihood_lose
    
    @property
    def posterior_lose(self):
        return self.prior_team*self.likelihood_win
    
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
    def __init__(self, teams, result, p_draw=0.0):
        if len(teams) != len(result): raise ValueError("len(teams) != len(result)")
        if (0.0 > p_draw) or (1.0 <= p_draw): raise ValueError ("0.0 <= proba < 1.0")
    
        self.teams = teams
        self.result = result
        self.p_draw = p_draw
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
    
    def likelihood_analitico(self):
        g = self
        r = g.result
        o = sortperm(r) 
        t = [g.performance(o[e]) for e in range(len(g))]
        d = [t[e] - t[e+1] for e in range(len(g)-1)]
        
        g.evidence = cdf(margin[e], d[e].mu, d[e].sigma)-cdf(-margin[e], d[e].mu, d[e].sigma) 
        
        def vt(t):
            global sqrt2pi, sqrt2
            pdf = (1 / sqrt2pi) * math.exp(- (t*t / 2))
            cdf = 0.5*(1+math.erf(t/ sqrt2))
            return (pdf/cdf)
        def wt(t, v):
            w = v * (v + t)
            return w
        delta = d[0].mu
        theta_pow2 = d[0].sigma**2
        theta = d[0].sigma
        t = delta/theta
        V = vt(t)
        W = wt(t, v=V)
        delta_div = (theta/(V+t))+delta
        theta_div_pow2 = (1/W-1)*theta_pow2
        def winner(mu_i, sigma_i, beta_i, delta=delta,
                   theta_pow2=theta_pow2, delta_div=delta_div,
                   theta_div_pow2=theta_div_pow2):
            mu = delta_div + mu_i - delta
            sigma_analitico = math.sqrt(theta_div_pow2 + theta_pow2
                                        - sigma_i*sigma_i)
            #return Rating(mu=mu, sigma=sigma_analitico, beta=beta_i)
            return Gaussian(mu=mu, sigma=sigma_analitico)
        def looser(mu_i, sigma_i, beta_i, delta=delta,
                   theta_pow2=theta_pow2, delta_div=delta_div,
                   theta_div_pow2=theta_div_pow2):
            mu = delta + mu_i - delta_div
            sigma_analitico = math.sqrt(theta_div_pow2 + theta_pow2
                                        - sigma_i*sigma_i)
            #return Rating(mu=mu, sigma=sigma_analitico, beta=beta_i)
            return Gaussian(mu=mu, sigma=sigma_analitico)
        player_winners = []
        for j in range(len(g.teams[o[0]])):
            player_winners.append(winner(g.teams[o[0]][j].N.mu, g.teams[o[0]][j].N.sigma, g.teams[o[0]][j].beta))
        player_loosers = []
        for j in range(len(g.teams[o[1]])):
            player_loosers.append(looser(g.teams[o[1]][j].N.mu, g.teams[o[1]][j].N.sigma, g.teams[o[1]][j].beta))
        return [player_winners ,player_loosers] if o[0]<o[1] else [player_loosers,player_winners]
    
    def likelihood_teams(self):
        g = self 
        r = g.result
        o = sortperm(r) 
        t = [team_messages(g.performance(o[e]),Ninf, Ninf, Ninf) for e in range(len(g))]
        d = [diff_messages(t[e].prior - t[e+1].prior, Ninf) for e in range(len(g)-1)]
        tie = [r[o[e]]==r[o[e+1]] for e in range(len(d))]
        margin = [0.0 if g.p_draw==0.0 else compute_margin(g.p_draw, math.sqrt( sum([a.beta**2 for a in g.teams[o[e]]]) + sum([a.beta**2 for a in g.teams[o[e+1]]]) )) for e in range(len(d))] 
        g.evidence = 1.0
        for e in range(len(d)):
            mu, sigma = d[e].prior.mu, d[e].prior.sigma
            g.evidence *= cdf(margin[e],mu,sigma)-cdf(-margin[e],mu,sigma) if tie[e] else 1-cdf(margin[e],mu,sigma)
        step = (inf, inf); i = 0 
        while gr_tuple(step,1e-6) and (i < 10):
            step = (0., 0.)
            for e in range(len(d)-1):
                d[e].prior = t[e].posterior_win - t[e+1].posterior_lose
                d[e].likelihood = Gaussian(*trunc(d[e].prior.mu,d[e].prior.sigma,margin[e],tie[e]))/d[e].prior
                likelihood_lose = t[e].posterior_win - d[e].likelihood
                step = max_tuple(step,t[e+1].likelihood_lose.delta(likelihood_lose))
                t[e+1].likelihood_lose = likelihood_lose
            for e in range(len(d)-1,0,-1):
                d[e].prior = t[e].posterior_win - t[e+1].posterior_lose
                d[e].likelihood = Gaussian(*trunc(d[e].prior.mu,d[e].prior.sigma,margin[e],tie[e]))/d[e].prior
                likelihood_win = t[e+1].posterior_lose + d[e].likelihood
                step = max_tuple(step,t[e].likelihood_win.delta(likelihood_win))
                t[e].likelihood_win = likelihood_win
            i += 1
        if len(d)==1:
            d[0].prior = t[0].posterior_win - t[1].posterior_lose
            d[0].likelihood = Gaussian(*trunc(d[0].prior.mu,d[0].prior.sigma,margin[0],tie[0]))/d[0].prior
        t[0].likelihood_win = t[1].posterior_lose + d[0].likelihood
        t[-1].likelihood_lose = t[-2].posterior_win - d[-1].likelihood
        return [ t[o[e]].likelihood for e in range(len(t)) ] 
    
    def compute_likelihoods(self):
        if len(self.teams)>2:
            m_t_ft = self.likelihood_teams()
            self.likelihoods = [[ m_t_ft[e] - self.performance(e).exclude(self.teams[e][i].N) for i in range(len(self.teams[e])) ] for e in range(len(self))]
        else:
            self.likelihoods = self.likelihood_analitico()            
        
    @property
    def posteriors(self):
        return [[ self.likelihoods[e][i] * self.teams[e][i].N for i in range(len(self.teams[e]))] for e in range(len(self))]

#ta = [Rating(0,1),Rating(0,1),Rating(0,1)]
#tb = [Rating(0,1),Rating(0,1),Rating(0,1)]
#tc = [Rating(0,1),Rating(0,1),Rating(0,1)]
#td = [Rating(0,1),Rating(0,1),Rating(0,1)]
#time_tt = timeit.timeit(lambda: Game([ta,tb],[1,0]).posteriors, number=10000)/10000
#ta = [ts.Rating(0,1),ts.Rating(0,1),ts.Rating(0,1)]
#tb = [ts.Rating(0,1),ts.Rating(0,1),ts.Rating(0,1)]
#tc = [ts.Rating(0,1),ts.Rating(0,1),ts.Rating(0,1)]
#td = [ts.Rating(0,1),ts.Rating(0,1),ts.Rating(0,1)]
#time_ts = timeit.timeit(lambda: ts.rate([ta,tb],[1,0]), number=10000)/10000

#time_ts/time_tt

class Skill(object):
    def __init__(self, forward=Ninf, backward=Ninf, likelihood=Ninf, elapsed=0):
        self.forward = forward
        self.backward = backward
        self.likelihood = likelihood
        self.elapsed = elapsed

class Agent(object):
    def __init__(self, prior, message, last_time):
        self.prior = prior
        self.message = message
        self.last_time = last_time
    
    def receive(self, elapsed):
        if self.message != Ninf:
            res = agent.message.forget(agent.prior.gamma, elapsed) 
        else:
            res = agent.prior.N
        return res

def clean(agents,last_time=False):
    for a in agents:
        agents[a].message = Ninf
        if last_time:
            agents[a].last_time = -inf

class Item(object):
    def __init__(self,name,likelihood):
        self.name = name
        self.likelihood = likelihood

class Team(object):
    def __init__(self, items, output):
        self.items = items
        self.output = output
    
class Event(object):
    def __init__(self, teams, evidence):
        self.teams = teams
        self.evidence = evidence
    
def get_composition(events):
    return [ [[ it.name for it in t.items] for t in e.teams] for e in events]

def get_results(events):
    return [ [t.output for t in e.teams ] for e in events]

def compute_elapsed(last_time, actual_time):
    return 0 if last_time == -inf  else ( 1 if last_time == inf else (actual_time - last_time))

class Batch(object):
    def __init__(self, composition, results, time, agents=dict(), env=Environment()):
        if len(events)!= len(results): raise ValueError("len(events)!= len(results)")
        
        this_agents = set( [a for event in events for team in event for a in team ] )
        elapsed = dict([ (a,  compute_elapsed(agents[a].last_time, time) ) for a in this_agents ])
        skills = dict([ (a, Skill(agents[a].receive(elapsed[a]) ,Ninf ,Ninf , elapsed[a])) for a in this_agents  ])
        
        self.events = [Event([Team([Item(composition[e][t][a], Ninf) for a in range(le(composition[e][t])) ], results[e][t]  ) for t in range(len(composition[e])) ],0.0) for e in range(len(composition) )]
        self.results = results
        self.time = time
        
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
        self.agents = set( [a for event in events for team in event for a in team ] )
        self.partake = dict()
        for a in self.agents: self.partake[a] = dict()
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
                self.partake[a][t] = b
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
        if len(self.batches)==1: self.batches[0].convergence()
        return step, i
    def learning_curves(self):
        res = dict()
        for a in self.agents:
            res[a] = sorted([ (t, self.partake[a][t].posterior(a)) for t in self.partake[a]])
        return res
    

def dict_diff(old, new):
    step = (0., 0.)
    for a in old:
        step = max_tuple(step, old[a].delta(new[a]))
    return step


                
