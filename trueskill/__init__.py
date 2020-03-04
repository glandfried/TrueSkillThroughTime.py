from __future__ import absolute_import
print("""
   trueskill
   ~~~~~~~~~
   :copyright: (c) 2012-2016 by Heungsub Lee.
   :copyright: (c) 2019-2020 by Gustavo Landfried.
   :license: BSD, see LICENSE for more details.
""")
import time
#from scipy.stats import norm
from scipy.stats import entropy
import numpy as np
#import matplotlib.pyplot as plt
import copy
#import trueskill as ts
#env = ts.TrueSkill(draw_probability=0,tau=0)
from collections import defaultdict
from .mathematics import Gaussian
#from mathematics import Gaussian
import ipdb

BETA = 25/6
 
__all__ = [
    # TrueSkill objects
    'TrueSkill', 'Rating' , 'Skill', 'Synergy',
    # functions for the global environment
    #'rate', 'quality', 'rate_1vs1', 'quality_1vs1', 'expose',
    'global_env', 'setup',
    # default values
    'MU_PLAYER', 'SIGMA_PLAYER', 'BETA_PLAYER', 'TAU_PLAYER', 'DRAW_PROBABILITY',
    'MU_TEAMMATE', 'SIGMA_TEAMMATE', 'BETA_TEAMMATE', 'TAU_TEAMMATE'
    # draw probability helpers
    #'calc_draw_probability', 'calc_draw_margin',
    # deprecated features
    #'transform_ratings', 'match_quality', 'dynamic_draw_probability',
]

#: Default initial mean of players.
MU_PLAYER = 25.
#: Default initial standard deviation of players.
SIGMA_PLAYER = MU_PLAYER / 3
#: Default random noise of players' performances. 
BETA_PLAYER = SIGMA_PLAYER / 2
#: Default dynamic factor of players.
TAU_PLAYER = SIGMA_PLAYER / 100
#: Default draw probability of the game.
DRAW_PROBABILITY = .10
#: A basis to check reliability of the result.
DELTA = 0.0001
#: Default initial mean of teammates.
MU_TEAMMATE = 0.
#: Default initial standard deviation of teammates.
SIGMA_TEAMMATE = 1.
#: Default random noise of teammates' performances.
BETA_TEAMMATE = 1.
#: Default dynamic factor of teammates.
TAU_TEAMMATE = 1. / 100

MU_HANDICAP = 0.
SIGMA_HANDICAP = 25./3
BETA_HANDICAP = 0.25
TAU_HANDICAP = 1. / 100

class Skill(Gaussian):
    """ Skill belief distribution
    """
    def __init__(self, mu=None, sigma=None, env=None):
        if isinstance(mu, tuple):
            mu, sigma = mu
        elif isinstance(mu, Gaussian):
            mu, sigma = mu.mu, mu.sigma
        if mu is None:
            mu = global_env().mu_player
        if sigma is None:
            sigma = global_env().sigma_player
        self.env = env
        super(Skill, self).__init__(mu, sigma)
    
    @property
    def noise(self):
        return self.env.tau_player if not self.env is None else global_env().tau_player    
    
    @property
    def beta(self):
        return self.env.beta_player if not self.env is None else global_env().beta_player    
    
    @property
    def forget(self):
        self.modify(Gaussian(self,self.noise))
        return self
    
    @property
    def performance(self):
        return Gaussian(self,self.beta)
    
    def play(self):
        return np.random.normal(*self.performance)

    def posterior(self,other):
        return Skill(self*other,self.env)
        
    def __repr__(self):
        c = type(self)
        if self.env is None:
            args = ( c.__name__,*iter(self))
        else: 
            args = ('.'.join(["TrueSkill", c.__name__]),*iter(self))
        return '%s(mu=%.3f, sigma=%.3f)' % args
    
class Rating(Gaussian):
    def __init__(self, mu=None, sigma=None, env=None, beta=None, noise=None):
        if isinstance(mu, tuple):
            mu, sigma = mu
        elif isinstance(mu, Gaussian):
            mu, sigma = mu.mu, mu.sigma
        if mu is None:
            mu = global_env().mu_player
        if sigma is None:
            sigma = global_env().sigma_player
        self.env = env
        self.beta = beta
        self.noise = noise
        if self.beta is None:
            self.beta = self.env.beta_player if not self.env is None else global_env().beta_player    
        if self.noise is None:
            self.noise = self.env.tau_player if not self.env is None else global_env().tau_player    
        super(Rating, self).__init__(mu, sigma)
    
    @property
    def forget(self):
        self.modify(Gaussian(self,self.noise))
        return self
    
    @property
    def performance(self):
        return Gaussian(self,self.beta)
    
    def play(self):
        return np.random.normal(*self.performance)       
    
    def posterior(self,other):
        return Rating(self*other,self.env,self.beta,self.noise)
    
    def __repr__(self):
        c = type(self)
        if self.env is None:
            args = ( c.__name__,*iter(self))
        else: 
            args = ('.'.join(["TrueSkill", c.__name__]),*iter(self))
        return '%s(mu=%.3f, sigma=%.3f)' % args


class Handicap(Gaussian):
    """ Skill belief distribution
    """
    def __init__(self, mu=None, sigma=None, env=None):
        if isinstance(mu, tuple):
            mu, sigma = mu
        elif isinstance(mu, Gaussian):
            mu, sigma = mu.mu, mu.sigma
        if mu is None:
            mu = global_env().mu_player
        if sigma is None:
            sigma = global_env().sigma_player
        self.env = env
        super(Skill, self).__init__(mu, sigma)
    
    @property
    def noise(self):
        return self.env.tau_player if not self.env is None else global_env().tau_player    
    
    @property
    def beta(self):
        return self.env.beta_player if not self.env is None else global_env().beta_player    
    
    @property
    def forget(self):
        self.modify(Gaussian(self,self.noise))
        return self
    
    @property
    def performance(self):
        return Gaussian(self,self.beta)

    def posterior(self,other):
        return Handicap(self*other,self.env)
    
    def play(self):
        return np.random.normal(*self.performance)

    def __repr__(self):
        c = type(self)
        if self.env is None:
            args = ( c.__name__,*iter(self))
        else: 
            args = ('.'.join(["TrueSkill", c.__name__]),*iter(self))
        return '%s(mu=%.3f, sigma=%.3f)' % args



class Synergy(Gaussian):
    """ Synergy belief distribution
    """

    def __init__(self, mu=None, sigma=None, env=None):
        if isinstance(mu, tuple):
            mu, sigma = mu
        elif isinstance(mu, Gaussian):
            mu, sigma = mu.mu, mu.sigma
        if mu is None:
            mu = global_env().mu_teammate
        if sigma is None:
            sigma = global_env().sigma_teammate
        self.env = env
        super(Synergy, self).__init__(mu, sigma)

    @property
    def noise(self):
        return self.env.tau_teammate if not self.env is None else global_env().tau_teammate    
    
    @property
    def beta(self):
        return self.env.beta_teammate if not self.env is None else global_env().beta_teammate    
    
    @property
    def forget(self):
        self.modify(Gaussian(self,self.noise))
        return self
    
    @property
    def performance(self):
        return Gaussian(self,self.beta)
    
    def posterior(self,other):
        return Synergy(self*other,self.env)
        
    def play(self):
        return np.random.normal(*self.performance)

    def __repr__(self):
        c = type(self)
        if self.env is None:
            args = ( c.__name__,*iter(self))
        else: 
            args = ('.'.join(["TrueSkill", c.__name__]),*iter(self))
        return '%s(mu=%.3f, sigma=%.3f)' % args 

class Team(Gaussian):
    def __init__(self, skills=None, synergys=None):
        self.skills = skills
        self.synergys = synergys
        self.ratings = self.skills
        if not self.synergys is None: 
            self.ratings += self.synergys 
        mu = sum(map(lambda s: s.mu, self.ratings))
        sigma = np.sqrt(np.sum(list(map(lambda s: s.performance.sigma**2, self.ratings))) )
        super(Team, self).__init__(mu, sigma)
        
    @property
    def performance(self):#e=0
        return Gaussian(self.mu,self.sigma)
    
    def play(self):
        return np.random.normal(*self.performance)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self,key):
        return self.ratings[key]
    
    def exclude(self,key):
        return Gaussian(self.mu - self[key].mu, np.sqrt(self.sigma**2 - self[key].sigma**2))
    
    def __repr__(self):
        c = type(self)
        args = ( c.__name__,*iter(self))
        return '%s(mu=%.3f, sigma=%.3f)' % args 

class Game(object):
    
    def __init__(self, teams = None, results = None, names = None):
        
        if isinstance(teams[0], Team):
            self.teams = list(teams)
        else:   
            self.teams = [ Team(t) for t in teams]
        self.results = list(results)
        
        self.names = names
        
        teams_index_sorted = sorted(zip(self.teams, range(len(self.teams)), self.results), key=lambda x: x[2])
        teams_sorted , index_sorted, _ = list(zip(*teams_index_sorted )) 
        
        #self.s = teams_sorted 
        self.o = list(index_sorted)
        self.t =  teams_sorted #[ teams_sorted[e].performance for e in range(len(teams)) ]
        self.d = [ self.t[e]-self.t[e+1] for e in range(len(self.t)-1)]
        self.likelihood = self.compute_likelihood()
        self.posterior = self.compute_posterior()
        self.evidence = self.compute_evidence()
        self.last_posterior , self.last_likelihood, self.last_evidence = self.posterior , self.likelihood, self.evidence
        
    def sortTeams(self,teams,results):
        teams_result_sorted = sorted(zip(teams, results), key=lambda x: x[1])
        res = list(zip(*teams_result_sorted )) 
        return list(res[0]), list(res[1])
        
    def last_posterior_of(self,elem):
        return sum(self.last_posterior,[])[sum(self.index,[]).index(elem)] 
    
    def last_likelihood_of(self,elem):
        return sum(self.last_likelihood,[])[sum(self.index,[]).index(elem)] 
    
    
    @property
    def m_t_ft(self):
        
        #truncadas = 0
        
        def thisDelta(old,new):
            mu_old, sigma_old = old
            mu_new, sigma_new = new
            return max(abs(mu_old-mu_new),abs(sigma_old-sigma_new))
    
        team_perf_messages = [[self.t[e], Gaussian(), Gaussian() ] for e in range(len(self.t))]
        
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
                
                #start = time.time()
                diff_messages[i][1] = diff_messages[i][0].trunc/diff_messages[i][0]
                #end = time.time()
                #truncadas += end-start
                                
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
                
                #start = time.time()
                diff_messages[i][1] = diff_messages[i][0].trunc/diff_messages[i][0]
                #end = time.time()
                #truncadas += end-start
                
                delta = max(delta, thisDelta(d_old,np.prod(diff_messages[i])))   
                
                team_perf_messages[i][2] = (
                    np.prod(team_perf_messages[i+1])/team_perf_messages[i+1][1]
                    +
                    diff_messages[i][1]
                )
        
            k += 1
            #print(delta,k)
        if len(self.t)==2:
            i = 0
            diff_messages[0][0] = (
                    np.prod(team_perf_messages[i])  / team_perf_messages[i][2]  
                    - 
                    np.prod(team_perf_messages[i+1])/ team_perf_messages[i+1][1]
                )
            
            diff_messages[0][1] = diff_messages[i][0].trunc/diff_messages[i][0]
            
            team_perf_messages[i+1][1] = (
                    np.prod(team_perf_messages[i])/team_perf_messages[i][2]
                    - 
                    diff_messages[i][1]
                )
            team_perf_messages[i][2] = (
                    np.prod(team_perf_messages[i+1])/team_perf_messages[i+1][1]
                    - 
                    diff_messages[i][1]
                )
        
        team_perf_messages[0][2] = np.prod(team_perf_messages[1])/team_perf_messages[1][1]+diff_messages[0][1]
        
        i = len(self.t)-2
        team_perf_messages[i+1][1] = np.prod(team_perf_messages[i])/team_perf_messages[i][2]-diff_messages[i][1]
    
            
        res = [Gaussian() for i in range(len(self.t))]
        for i in range(len(self.t)):
            res[i] = np.prod(team_perf_messages[i])/team_perf_messages[i][0]
        
        #print ('trunc ', truncadas)
        return res
    
    def compute_likelihood(self):
        #start = time.time()
        t_ft = self.m_t_ft
        #end = time.time()
        #print(end - start, "m_t_ft")    
        return [[t_ft[e]- self.t[e].exclude(i) for i in range(len(self.t[e]))] for e in range(len(self.t))]
    
    def compute_posterior(self):    
        prior = self.t
        likelihood = self.likelihood
        res= [[prior[e][i].posterior(likelihood[e][i]) for i in range(len(self.t[e]))] for e in range(len(self.t))]    
        res, _ = self.sortTeams(res,self.o)
        return res 
    
    def compute_evidence(self):
        res = 1
        for d in self.d:
            res *= d.cdf(d.mu/d.sigma)
        return res
     
    def __iter__(self):
        return iter(self.t)

    def __repr__(self):
        c = type(self)
        return '{}({},{})'.format(c.__name__,self.teams,self.o)        
    
    def update(self,new_prior=None):
        """
        TERMINAR, para usar con forward y backward cuando new_prior is not None
        """
        if new_prior is None:
            return self.compute_posterior(), self.compute_likelihood(), self.compute_evidence()
        else:
            print("Implementar forward y backward TTT en metodos posterior y likelihood")
            
class History(object):
    def __init__(self, names, results, times=None):
        """
        names: list of composition [[1,2],[1,3]] or [[[1],[2]],[[1,2],[3]]]
        results : list of result
        times : list of time
        """
        self.names = names
        self.results = results
        self.times = times
        self.players = defaultdict(lambda: th.Skill(mu=25,sigma=25/3))
        self.games = []
    
    def __len__(self):
        return len(self.names)
    
    """
    TODO: Seguir!!!!!!!!
    Ver test.py
    """
    
    
    """
    def create_games(self):
        for g in range(len(self)):#g=0
            teams = [[players[i] for i in ti ] if isinstance(ti, list) else [players[i]] for ti in names[g] ]
            history.append(th.Game(teams,list_results[g]))
            update_prior(sum(list_teams[g],[]) ,history[-1].posterior,prior)
        return history, prior
    """  
    
    
class TrueSkill(object):
    def __init__(self, mu_player=MU_PLAYER, sigma_player=SIGMA_PLAYER
                 , beta_player=BETA_PLAYER, tau_player = TAU_PLAYER
                 , mu_teammate=MU_TEAMMATE, sigma_teammate=SIGMA_TEAMMATE
                 , beta_teammate=BETA_TEAMMATE, tau_teammate = TAU_TEAMMATE
                 , mu_handicap=MU_HANDICAP, sigma_handicap=SIGMA_HANDICAP
                 , beta_handicap=BETA_HANDICAP, tau_handicap= TAU_HANDICAP
                 , draw_probability=DRAW_PROBABILITY):
        # Player
        self.mu_player =mu_player 
        self.sigma_player=sigma_player
        self.beta_player=beta_player
        self.tau_player=tau_player
        # Teammate
        self.mu_teammate =mu_teammate 
        self.sigma_teammate=sigma_teammate
        self.beta_teammate=beta_teammate  
        self.tau_teammate=tau_teammate
        # others
        self.draw_probability = draw_probability
    
    
    def rating(self, mu=None, sigma=None, beta=None, noise=None):
        if mu is None:
            mu = self.mu_player
        if sigma is None:
            sigma = self.sigma_player
        return Rating(mu,sigma,self,beta,noise)
        
    def skill(self, mu=None, sigma=None):
        """Initializes new :class"""
        if mu is None:
            mu = self.mu_player
        if sigma is None:
            sigma = self.sigma_player
        return Skill(mu, sigma, self)   
    
    def synergy(self, mu=None, sigma=None):
        """Initializes new :class:"""
        if mu is None:
            mu = self.mu_teammate
        if sigma is None:
            sigma = self.sigma_teammate
        return Synergy(mu, sigma, self)   
    
    
    def team(self, players= None, teammates=None):
        skills = [ self.Skill(*s) for s in players]
        synergys = [ self.Synergy(*s) for s in teammates] if not teammates is None else None
        return Team(skills,synergys )
    
    def game(self, teams=[], results=[]):
        return Game(teams,results)
    
    @property
    def Rating(self):
        return self.rating
    
    @property
    def Skill(self):
        return self.skill
    
    @property
    def Synergy(self):
        return self.synergy
    
    @property
    def Team(self):
        return self.team
        
    @property
    def Game(self):
        return self.game
    
    def rate(self,teams,results):
        _teams = [ self.Team(t) for t in teams]
        g = self.Game(_teams,results)
        return g.posterior
        
  
    def __iter__(self):
        return iter((self.mu_player, self.sigma_player, self.beta_player,self.tau_player
                     ,self.mu_teammate,self.sigma_teammate,self.beta_teammate,self.tau_teammate
                     ,self.draw_probability))

    
    def __repr__(self):
        c = type(self)
        draw_probability = '%.1f%%' % (self.draw_probability * 100)
        args = ('.'.join([c.__module__, c.__name__]), *iter(self), draw_probability)
        return ('%s(mu_player=%.3f, sigma_player=%.3f, beta_player=%.3f, tau_player=%.3f, '
                'mu_teammate=%.3f, sigma_teammate=%.3f, beta_teammate=%.3f, tau_teammate=%.3f, '    
                'draw_probability=%s%s)' % args)
    

def global_env():
    """Gets the :class:`TrueSynergy` object which is the global environment."""
    try:
        global_env.__truesynergy__
    except AttributeError:
        # setup the default environment
        setup()
    return global_env.__truesynergy__

def setup(mu_player=MU_PLAYER, sigma_player=SIGMA_PLAYER
          , beta_player=BETA_PLAYER, tau_player = TAU_PLAYER
          , mu_teammate=MU_TEAMMATE, sigma_teammate=SIGMA_TEAMMATE
          , beta_teammate=BETA_TEAMMATE, tau_teammate = TAU_TEAMMATE
          , draw_probability=DRAW_PROBABILITY, env=None):
    """Setups the global environment."""
    if env is None:
        env = TrueSkill(mu_player, sigma_player, beta_player, tau_player
                         ,mu_teammate, sigma_teammate, beta_teammate, tau_teammate
                         ,draw_probability)
    global_env.__truesynergy__ = env
    return env

def kl(p,q):
    return entropy(p,q)
        
    
