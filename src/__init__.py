# -*- coding: utf-8 -*-
from __future__ import absolute_import
print("""
   Trueskill
   ~~~~~~~~~
   :copyright: (c) 2012-2016 by Heungsub Lee.
   :copyright: (c) 2019-2020 by Gustavo Landfried.
   :license: BSD, see LICENSE for more details.
   
   Trueskill Through Time
   ~~~~~~~~~
   :copyright: (c) 2019-2020 by Gustavo Landfried.
   :license: BSD, see LICENSE for more details.
""")

"""
Complejidad computacional:
    0. TTT tarda al menos k*t*2*partidas*(0.003*(1/0.76)). 
       La complejidad es O(k*partidas), la t puede acotarse.
    1. El 76% del tiempo se consume en crear partidas.
    2. Cada partida (de dos equiopos) tarda ~0.003 segundos en crearse.
    3. Cada convergencia requiere dos ciclos (forward+backward)
    4. Luego, por cada convergencia:
       - 2.4 (3.16) segundos con 400 partidas por convergencia
       - 1.67 (2.19) horas con 1M partidas por convergencia
    5. La cantidad de convergencias k: ¿de qu\'e depende?
    6. La t es cantidad de convergencia al interiior de los tiempos
Objetivos:
    - Ir agregando partidas de tiempos
    - Implementar evidencia de a tiempos
"""

from scipy.stats import entropy
import numpy as np
#from datetime import datetime
#from datetime import timedelta
from collections import defaultdict
from collections.abc import Iterable
from .mathematics import Gaussian
#from mathematics import Gaussian
import time as clock
#import ipdb

BETA = 25/6
 
__all__ = [
    'TrueSkill', 'Rating' ,'Team', 'Game', 'History',
    'global_env', 'setup',
    'MU', 'SIGMA', 'BETA', 'TAU', 'DRAW_PROBABILITY', 'EPSILON'
]

#: Default initial mean 
MU = 25.
#: Default initial standard deviation
SIGMA = MU / 3
#: Default random noise
BETA = SIGMA / 2
#: Default dynamic factor
TAU = SIGMA / 100
#: Default draw probability of the game.
DRAW_PROBABILITY = .10
#: Epsilon
EPSILON = 1e-1
 
class Rating(Gaussian):
    def __init__(self, mu=None, sigma=None, env=None, beta=None, noise=None):
        self.env = global_env() if env is None  else env
        if isinstance(mu, tuple):  mu, sigma = mu
        elif isinstance(mu, Gaussian): mu, sigma = mu.mu, mu.sigma
        mu = self.env.mu if mu is None else mu
        sigma = self.env.sigma if sigma is None else sigma
        self.beta = self.env.beta if beta is None else beta
        self.noise = self.env.tau if noise is None else noise
        super(Rating, self).__init__(mu, sigma)
    
    def forget(self,t=1):
        new = Gaussian(self,self.noise*t)
        return Rating(new.mu, min(new.sigma,self.env.sigma),self.env,self.beta,self.noise)
    
    @property
    def performance(self):
        return Gaussian(self,self.beta)
    
    def play(self):
        return np.random.normal(*self.performance)       
    
    def filtered(self,other):
        res = self*other
        return Rating(res.mu, res.sigma,self.env,self.beta,self.noise)
    
    def inherit(self,other):
        return Rating(other.mu, other.sigma,self.env, self.beta, self.noise)
    
    def __repr__(self):
        c = type(self)
        if self.env is None:
            args = ( c.__name__,*iter(self),self.beta,self.noise)
        else: 
            args = ('.'.join(["TrueSkill", c.__name__]),*iter(self),self.beta,self.noise)
        return '%s(mu=%.3f, sigma=%.3f, beta=%.3f, noise=%.3f)' % args

class Team(object):
    def __init__(self, ratings=None):
        self.ratings = ratings
        self.prior = self.ratings
        self.inverse_prior = [Gaussian() for _ in self.ratings]
        #self.old_likelihood = [ [ Gaussian() for i in te] for te in self.ratings ]
        #super(Team, self).__init__(self.mu, self.sigma)
            
    @property
    def mu(self):
        return sum(map(lambda s: s.mu, self.ratings))
    @property
    def sigma(self):
        return np.sqrt(np.sum(list(map(lambda s: s.performance.sigma**2, self.ratings))) )
    
    @property
    def performance(self):
        return np.sum([ (self.prior[i].performance*self.inverse_prior[i]) for i in range(len(self)) ]) 
    
    def back_info(self,inverse_prior):
        self.inverse_prior = inverse_prior
    
    def for_info(self,prior):
        self.prior = prior
        self.ratings = prior
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self,key):
        return self.ratings[key]
    
    def exclude(self,key):
        mu = self.mu - self[key].mu
        sigma = np.sqrt(self.sigma**2 - self[key].sigma**2)
        #ipdb.set_trace()
        return Gaussian(mu,sigma)
    
    def __repr__(self):
        res = 'Team('
        res += '{}'.format(self.ratings[0])
        for r in range(1,len(self.ratings)):
            res += ', '
            res += '{}'.format(self.ratings[r])
        res += ')'
        return res

class Game(object):
    
    def __init__(self, teams = None, results = None, names = None):
        
        if isinstance(teams[0], Team):
            self.teams = list(teams)
        else:   
            self.teams = [ Team(t) for t in teams]
        self.results = list(results)
        if not names is None:
            self.names = [list(n) if isinstance(n,Iterable) else [n] for n in names]
            
        teams_index_sorted = sorted(zip(self.teams, range(len(self.teams)), self.results), key=lambda x: x[2])
        teams_sorted , index_sorted, _ = list(zip(*teams_index_sorted )) 
    
        self.o = list(index_sorted)
        self.t = teams_sorted
        self.d = [ self.t[e].performance - self.t[e+1].performance for e in range(len(self.t)-1)]
        
        #self.likelihood = [[Gaussian() for r in te.ratings] for te in self.teams]
        #self.prior = [[r for r in te.ratings] for te in self.teams] 
        #self.inverse_prior = [[Gaussian() for r in te.ratings] for te in self.teams] 
        

        self.likelihood = self.compute_likelihood() # 0.0025
        self.posterior = self.compute_posterior() # 0.00025
        self.evidence = self.compute_evidence() # 0.00006
        
    def sortTeams(self,teams,results):
        teams_result_sorted = sorted(zip(teams, results), key=lambda x: x[1])
        res = list(zip(*teams_result_sorted )) 
        return list(res[0]), list(res[1])
    
    def prior(self):
        return [te.ratings for te in self.teams]
    def inverse_prior(self):
        return [te.inverse_prior for te in self.teams]
    
    @property
    def m_t_ft(self):
        
        #truncadas = 0
        
        def thisDelta(old,new):
            mu_old, sigma_old = old
            mu_new, sigma_new = new
            return max(abs(mu_old-mu_new),abs(sigma_old-sigma_new))
        
        team_perf_messages = [[self.t[e].performance, Gaussian(), Gaussian() ] for e in range(len(self.t))]
        
        diff_messages = [ [Gaussian(), Gaussian()] for i in range(len(self.t)-1) ]
        
        k=0; delta=np.inf
        while delta > 1e-4:
            delta = 0
            for i in range(len(self.t)-2):#i=0
                d_old = np.prod(diff_messages[i])
                
                diff_messages[i][0] = (
                    np.prod(team_perf_messages[i])  / team_perf_messages[i][2]  
                    - 
                    np.prod(team_perf_messages[i+1])/ team_perf_messages[i+1][1]
                )
                
                diff_messages[i][1] = diff_messages[i][0].trunc/diff_messages[i][0]
                                
                delta = max(delta, thisDelta(d_old,np.prod(diff_messages[i])))   
                
                team_perf_messages[i+1][1] = (
                    np.prod(team_perf_messages[i])/team_perf_messages[i][2]
                    - 
                    diff_messages[i][1]
                )
                
            for i in range(len(self.t)-2,0,-1):#i=8
                d_old = np.prod(diff_messages[i])
                
                diff_messages[i][0] = (
                    np.prod(team_perf_messages[i]) / team_perf_messages[i][2]
                    -
                    np.prod(team_perf_messages[i+1])/team_perf_messages[i+1][1]
                )
                
                diff_messages[i][1] = diff_messages[i][0].trunc/diff_messages[i][0]
                
                delta = max(delta, thisDelta(d_old,np.prod(diff_messages[i])))   
                
                team_perf_messages[i][2] = (
                    np.prod(team_perf_messages[i+1])/team_perf_messages[i+1][1]
                    +
                    diff_messages[i][1]
                )
        
            k += 1
            
        if len(self.t)==2:
            i = 0
            diff_messages[0][0] = (
                    np.prod(team_perf_messages[i])  / team_perf_messages[i][2]  
                    - 
                    np.prod(team_perf_messages[i+1])/ team_perf_messages[i+1][1]
                )
            
            diff_messages[0][1] = diff_messages[i][0].trunc/diff_messages[i][0]
        
        #ipdb.set_trace()
        team_perf_messages[0][2] = np.prod(team_perf_messages[1])/team_perf_messages[1][1]+diff_messages[0][1]
        
        i = len(self.t)-2
        team_perf_messages[i+1][1] = np.prod(team_perf_messages[i])/team_perf_messages[i][2]-diff_messages[i][1]
    
            
        res = [Gaussian() for i in range(len(self.t))]
        for i in range(len(self.t)):
            res[i] = np.prod(team_perf_messages[i])/team_perf_messages[i][0]
        
        return res

    @property
    def m_fp_s(self):
        t_ft = self.m_t_ft
        return [[t_ft[e]- self.t[e].exclude(i) for i in range(len(self.t[e]))] for e in range(len(self.t))]
    
    def compute_likelihood(self):
        likelihood, _ = self.sortTeams(self.m_fp_s,self.o)
        return likelihood
    
    def compute_posterior(self):    
        prior = self.teams
        likelihood = self.likelihood
        posterior = [[prior[e][i].filtered(likelihood[e][i]) for i in range(len(prior[e]))] for e in range(len(prior))]    
        return posterior 
    
    def compute_evidence(self):
        res = 1
        for d in self.d:
            res *= 1-d.cdf(0)
        return res
    
    @property
    def update(self):
        #ipdb.set_trace()
        self.last_posterior = self.compute_posterior()
        self.last_likelihood = self.compute_likelihood() 
        self.last_evidence = self.compute_evidence()
        return self.last_likelihood, self.last_posterior
        
    def __iter__(self):
        return iter(self.t)

    def __getitem__(self,key):
        return self.teams[key]
    
    def __len__(self):
        return len(self.teams)
    
    def __repr__(self):
        c = type(self)
        return '{}({},{})'.format(c.__name__,self.teams,self.o)

class Time(object):
    def __init__(self,games_composition,results,forward_priors=defaultdict(lambda: Rating()) 
    , batch_number=None,last_batch=defaultdict(lambda: None),match_id=None,epsilon=1e-2):
        """
        games_composition = [[[1],[2]],[[1],[3]],[[2],[3]]]
        """
        self.games_composition = games_composition
        self.results = results
        self.batch_number = batch_number
        self.match_id = match_id
        self.players = set(flat(flat(games_composition) ))
        self.played = {}
        for i in self.players:
            self.played[i] = self.games_played(i) 
        self.time_elapsed = {}
        for i in self.players:
            elapsed = 0 if last_batch[i] is None else last_batch[i] - batch_number
            self.time_elapsed[i] = elapsed  
        
        self.forward_priors = {}
        for i in self.players:
            self.forward_priors[i] = forward_priors[i].forget(self.time_elapsed[i])
        
        self.priors = dict(self.forward_priors)
        self.backward_priors = defaultdict(lambda: Gaussian())
        self.likelihoods = defaultdict(lambda: defaultdict(lambda: Gaussian()))    
        self.epsilon = epsilon
        self.game_time = 0
        self.evidence = []
        self.last_evidence = []
        self.convergence()
        self.match_evidence = dict(zip(self.match_id,self.evidence))
        self.match_last_evidence = None
    
    
    def __len__(self):
        return len(self.games_composition)
    
    def games_played(self,i):
        res = []
        for g in range(len(self)):
            if i in flat(self.names(g)): res.append(g)
        return set(res)
    
    def time_likelihood(self,i):
        return np.prod([ self.likelihoods[i][k]  for k in self.played[i] ])
    
    def forward_posterior(self,i):
        return self.forward_priors[i].filtered(self.time_likelihood(i))
     
    def backward_posterior(self,i):
        res = self.time_likelihood(i)*self.backward_priors[i]
        return self.forward_priors[i].inherit(res)
    
    def posterior(self,i):
        return self.forward_priors[i].filtered(self.time_likelihood(i)*self.backward_priors[i])
    
    @property
    def forward_priors_out(self):
        res = {}
        for i in self.players:
            res[i] = self.forward_posterior(i).forget(0)
        return res

    @property
    def backward_priors_out(self):
        res = {}
        for i in self.players:
            res[i] = self.backward_posterior(i).forget(self.time_elapsed[i])
        return res    
        
    @property
    def posteriors(self):
        res = {}
        for i in self.players:
            res[i] = self.posterior(i)
        return res
    
    def within_prior(self,i,g):
        return self.forward_priors[i].filtered( (self.time_likelihood(i)/self.likelihoods[i][g]) * self.backward_priors[i] )
    
    def within_priors(self,g):
        teams = self.games_composition[g]
        return [ [self.within_prior(i,g) for i in te] for te in teams]

    def teams(self,g):
        return len(self.games_composition[g])
    
    def names(self,g):
        return self.games_composition[g]

    def iteration(self):
        max_delta = 0
        for g in range(len(self)):
            within_priors = self.within_priors(g)
            game_start = clock.time()
            game = Game(within_priors,self.results[g],self.games_composition[g])
            game_end = clock.time()
            self.game_time +=  game_end - game_start
            for te in range(self.teams(g)):
                names = self.names(g)[te]
                for i in range(len(names)):
                    n = names[i] 
                    max_delta = max(abs(game.likelihood[te][i].mu - self.likelihoods[n][g].mu),max_delta)
                    self.likelihoods[n][g] = game.likelihood[te][i]
            if not (len(self.evidence) > g):
                """
                TODO: 
                    la evidencia debe ser calculada con el forward_prior,
                    no con el within prior.
                """
                self.evidence.append(game.evidence)
                self.last_evidence.append(game.evidence)
            else:
                #ipdb.set_trace()
                self.last_evidence[g] = game.evidence
        return max_delta
            
    def convergence(self):
        delta = np.inf
        iterations = 0
        while delta > self.epsilon and iterations < 10:
            delta = self.iteration()
            #print(delta)
            iterations += 1
        #print(iterations)
        self.match_last_evidence = dict(zip(self.match_id,self.last_evidence))
        return iterations
    
    def backward_info(self,backward_priors):
        for i in self.players:
            self.backward_priors[i] = backward_priors[i]
        return self.convergence()
        
    def forward_info(self,forward_priors):
        for i in self.players:
            self.forward_priors[i] = forward_priors[i].forget(self.time_elapsed[i])
        return self.convergence()
        
    def __repr__(self):
        c = type(self)
        return '{}{}'.format(c.__name__,tuple(zip(self.games_composition,self.results)))

class History(object):
    def __init__(self , games_composition , results , batch_numbers=None
                 , prior_dict = {} , default=None  , match_id = None , epsilon=10**-3, env=None):
        
        self.env = global_env() if env is None  else env
        self.games_composition = list(map(lambda xs: xs if isinstance(xs[0],list) else [ [x] for x in xs] ,games_composition))
        self.results = results
        self.batch_numbers = batch_numbers
        self.match_id = list(range(len(self))) if match_id is None else match_id
        if not self.batch_numbers is None:
            self.games_composition, self.results, self.batch_numbers, self.match_id = map(lambda x: list(x),list(zip(*sorted(zip(self.games_composition,self.results, self.batch_numbers,self.match_id), key=lambda x: x[2]))))  
        self.default = env.Rating() if default is None else default
        self.epsilon = epsilon
                
        self.initial_prior= defaultdict(lambda: self.default)
        self.initial_prior.update(prior_dict)
        self.forward_priors = self.initial_prior.copy()
        self.backward_priors  = defaultdict(lambda: Gaussian())
        self.times = []
        self.match_time = {}

        self.last_batch = defaultdict(lambda: None)
        
        ############
        # Por si queremos comparar con trueskill
        self.forward_priors_trueskill = self.initial_prior.copy()
        self.times_trueskill = []
        ########
        
        self.learning_curves_trueskill = {}
        self.learning_curves_online = defaultdict(lambda: []) 
        self.learning_curves = {}
        
        #self.trueSkill()
        #self.through_time()
        #self.through_time(online=False); self.convergence()    
    
    def end_batch(self,i):
        t = None if self.batch_numbers is None else self.batch_numbers[i]
        j = i + 1
        while (j < len(self)) and (not t is None) and (self.batch_numbers[j] <= t):
            j += 1
        return j
    
        
    def trueSkill(self):
        i = 0
        while i < len(self):
            t = None if self.batch_numbers is None else self.batch_numbers[i]
            j = self.end_batch(i)
            time = Time(games_composition = self.games_composition[i:j]
                        ,results = self.results[i:j]
                        ,forward_priors = self.forward_priors_trueskill
                        ,batch_number = t
                        ,epsilon = self.epsilon
            )        
            self.forward_priors_trueskill.update(time.forward_priors_out)
            self.times_trueskill.append(time)
            i = j 
        self.update_learning_curve_trueskill()
            
    def update_learning_curve_trueskill(self):
        self.learning_curves_trueskill = defaultdict(lambda: []) 
        for time in self.times_trueskill:
            for i in time.posteriors:
                self.learning_curves_trueskill[i].append(time.posteriors[i])

    def delta(self,new,old):
        return np.max([ abs(new[i].mu - old[i].mu) for i in old])
    
    def backward_propagation(self):
        self.backward_priors  = defaultdict(lambda: Gaussian()) 
        delta = 0
        for t in reversed(range(len(self.times)-1)):
            self.backward_priors.update(self.times[t+1].backward_priors_out)
            old = self.times[t].posteriors
            self.times[t].backward_info(self.backward_priors)
            new = self.times[t].posteriors
            delta = max(self.delta(new,old),delta)
        return delta
    
    def forward_propagation(self):
        self.forward_priors = self.initial_prior.copy()
        delta = 0
        for t in range(1,len(self.times)):
            self.forward_priors.update(self.times[t-1].forward_priors_out)
            old = self.times[t].posteriors
            self.times[t].forward_info(self.forward_priors)
            new = self.times[t].posteriors
            delta = max(self.delta(new,old),delta)
            #print('Porcentaje:', int(t/len(self)*1000), t, delta,  end='\r')
        return delta
    
    def update_learning_curves(self):
        self.learning_curves = defaultdict(lambda: []) 
        for time in self.times:
            for i in time.posteriors:
                self.learning_curves[i].append(time.posteriors[i])

            
    def through_time(self,online=True):
        i = 0
        #ipdb.set_trace()
        print("Start first pass")
        start = clock.time()
        while i < len(self):
            t = 1 if self.batch_numbers is None else self.batch_numbers[i]
            j = self.end_batch(i)
            time = Time(games_composition = self.games_composition[i:j]
                        ,results = self.results[i:j]
                        ,forward_priors = self.forward_priors
                        ,batch_number = t
                        ,last_batch = self.last_batch
                        ,match_id = self.match_id[i:j]
                        ,epsilon = self.epsilon
            )
            self.match_time.update({m:time for m in self.match_id[i:j]})
            if not self.batch_numbers is None:
                self.last_batch.update(dict([(p,t) for p in time.players]))
            else:
                self.last_batch.update(dict([(p,0) for p in time.players]))
            self.times.append(time)
            if online:
                self.convergence()
            self.forward_priors.update(time.forward_priors_out)
            for i in time.posteriors:
                self.learning_curves_online[i].append(time.posteriors[i])
            i = j
        end = clock.time()
        print("End first pass:", round(end-start,3))
    
    def convergence(self):
        delta = np.inf
        for i in range(10):
            start = clock.time()
            delta = min(self.backward_propagation(),delta)            
            delta = min(self.forward_propagation(),delta)
            end = clock.time()
            print("d: ",round(delta,6),", t: ",round(end-start,4))#, end='\r')
            if delta < self.epsilon: break
        self.update_learning_curves()
    
    def players(self):
        return set(flat(flat(self.games_composition)))
    
    def individual_evidence(self,which=None):
        res = defaultdict(lambda: [])
        if which == 'TrueSkill':
            times = self.times_trueskill
        else:
            times = self.times
        for t in times:
            for g in range(len(t)):
                comp = t.games_composition[g]
                for e in range(len(comp)):
                    for i in range(len(comp[e])):
                        if which == 'TTT':
                            res[comp[e][i]].append(t.last_evidence[g])
                        else:
                            res[comp[e][i]].append(t.evidence[g])  
        return res
    
    def log10_evidence(self):
        return np.sum(np.log10(flat(list(map(lambda t: t.last_evidence, self.times )))))
    
    def log10_online_evidence(self):
        return np.sum(np.log10(flat(list(map(lambda t: t.evidence, self.times )))))
        
    def log10_evidence_trueskill(self):
        return np.sum(np.log10(flat(list(map(lambda t: t.evidence, self.times_trueskill )))))
    
    def __len__(self):
        return len(self.games_composition)
    
class TrueSkill(object):
    def __init__(self, mu=MU, sigma=SIGMA, beta=BETA, tau = TAU
                 , draw_probability=DRAW_PROBABILITY, epsilon=EPSILON):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.tau = tau
        self.draw_probability = draw_probability
        self.epsilon = epsilon
    
    def rating(self, mu=None, sigma=None, beta=None, noise=None):
        if mu is None: mu = self.mu
        if sigma is None: sigma = self.sigma
        if beta is None: beta = self.beta
        if noise is None: noise = self.tau
        return Rating(mu,sigma,self,beta,noise)
        
    def team(self, members):
        te = [ self.rating(mu=m.mu,sigma=m.sigma,beta=m.beta,noise=m.noise) for m in members]
        return Team(te)
    
    def game(self, teams=[], results=[], names=[]):
        return Game(teams,results,names)
    
    def history(self,games_composition,results, batch_numbers=None, prior_dict = {},epsilon=None ):
        if epsilon is None: epsilon = self.epsilon
        return History(games_composition, results , batch_numbers, prior_dict           
                 , default=self.Rating()
                 , epsilon=epsilon)
        
    @property
    def Rating(self):
        return self.rating

    @property
    def Team(self):
        return self.team
        
    @property
    def Game(self):
        return self.game
    
    @property
    def History(self):
        return self.history
    
    def rate(self,teams,results):
        _teams = [ self.Team(t) for t in teams]
        g = self.Game(_teams,results)
        return g.posterior
    
    def make_as_global(self):
        return setup(env=self)
  
    def __iter__(self):
        return iter((self.mu, self.sigma, self.beta,self.tau,self.draw_probability))

    
    def __repr__(self):
        c = type(self)
        draw_probability = '%.1f%%' % (self.draw_probability * 100)
        args = ('.'.join([c.__module__, c.__name__]), *iter(self), draw_probability)
        return ('%s(mu=%.3f, sigma=%.3f, beta=%.3f, tau=%.3f, draw_probability=%s%s)' % args)
    

def global_env():
    """Gets the :class:`TrueSynergy` object which is the global environment."""
    try:
        global_env.__truesynergy__
    except AttributeError:
        # setup the default environment
        setup()
    return global_env.__truesynergy__

def setup(mu=MU, sigma=SIGMA, beta=BETA, tau=TAU, draw_probability=DRAW_PROBABILITY, env=None):
    """Setups the global environment."""
    if env is None:
        env = TrueSkill(mu, sigma, beta, tau, draw_probability)
    global_env.__truesynergy__ = env
    return env

def kl(p,q):
    return entropy(p,q)
  
def flat(xs):
    if len(xs) == 0 or not isinstance(xs[0],list):
        res = xs
    else:
        res = sum(xs,[])
    return res
      
    
