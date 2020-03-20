from __future__ import absolute_import
print("""
   trueskill
   ~~~~~~~~~
   :copyright: (c) 2012-2016 by Heungsub Lee.
   :copyright: (c) 2019-2020 by Gustavo Landfried.
   :license: BSD, see LICENSE for more details.
""")
from scipy.stats import entropy
import numpy as np
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
from collections.abc import Iterable
from .mathematics import Gaussian
#from mathematics import Gaussian
import ipdb

BETA = 25/6
 
__all__ = [
    'TrueSkill', 'Rating' , 'Skill', 'Synergy',
    'global_env', 'setup',
    'MU_PLAYER', 'SIGMA_PLAYER', 'BETA_PLAYER', 'TAU_PLAYER', 'DRAW_PROBABILITY',
    'MU_TEAMMATE', 'SIGMA_TEAMMATE', 'BETA_TEAMMATE', 'TAU_TEAMMATE'
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

    def filtered(self,other):
        return Skill(self*other,self.env)
    
    #def modify(self, other):
    #    super(Skill, self).modify(other)
    
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
    
    def filtered(self,other):
        res = self*other
        return Rating(res.mu, res.sigma,self.env,self.beta,self.noise)
    
    def __repr__(self):
        c = type(self)
        if self.env is None:
            args = ( c.__name__,*iter(self),self.beta)
        else: 
            args = ('.'.join(["TrueSkill", c.__name__]),*iter(self),self.beta)
        return '%s(mu=%.3f, sigma=%.3f, beta=%.3f)' % args

class Team(object):
    def __init__(self, skills=None, synergys=None):
        self.skills = skills
        self.synergys = synergys
        self.ratings = self.skills
        if not self.synergys is None: 
            self.ratings += self.synergys 
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
        return Gaussian(self.mu - self[key].mu, np.sqrt(self.sigma**2 - self[key].sigma**2))
    
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
        #self.teams = [ list(t) if isinstance(t,Iterable) else [t]  for t in teams]
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
        
        self.vanilla_likelihood = self.compute_likelihood()
        self.vanilla_posterior = self.compute_posterior()
        self.vanilla_evidence = self.compute_evidence()
        self.likelihood = self.vanilla_likelihood
        self.posterior = self.vanilla_posterior 
        self.evidence = self.vanilla_evidence 
        
        self.last_posterior, self.last_likelihood, self.last_evidence = self.vanilla_posterior , self.vanilla_likelihood, self.vanilla_evidence
    
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

    @property
    def m_fp_s(self):
        #start = time.time()
        t_ft = self.m_t_ft
        #end = time.time()
        #print(end - start, "m_t_ft")    
        return [[t_ft[e]- self.t[e].exclude(i) for i in range(len(self.t[e]))] for e in range(len(self.t))]
    
    def compute_likelihood(self):
        likelihood, _ = self.sortTeams(self.m_fp_s,self.o)
        return likelihood
    
    def compute_posterior(self):    
        prior = self.t
        likelihood = self.m_fp_s
        res= [[prior[e][i].filtered(likelihood[e][i]) for i in range(len(self.t[e]))] for e in range(len(self.t))]    
        posterior, _ = self.sortTeams(res,self.o)
        return posterior 
    
    def compute_evidence(self):
        res = 1
        for d in self.d:
            res *= d.cdf(d.mu/d.sigma)
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
    def __init__(self,games_composition,results,forward_priors=defaultdict(lambda: Rating()) , time_step=None,epsilon=10**-3):
        """
        games_composition = [[[1],[2]],[[1],[3]],[[2],[3]]]
        """
        self.games_composition = games_composition
        self.results = results
        self.time_step = time_step
        self.players = set(flat(flat(games_composition) ))
        self.played = {}
        for i in self.players:
            self.played[i] = self.games_played(i) 
        self.forward_priors = {}
        for i in self.players:
            self.forward_priors[i] = forward_priors[i]
        self.backward_priors = defaultdict(lambda: Gaussian())
        self.likelihoods = defaultdict(lambda: defaultdict(lambda: Gaussian()))    
        self.epsilon = epsilon
        self.convergence()
    
    
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
        return self.time_likelihood(i)*self.forward_priors[i]
    
    def backward_posterior(self,i):
        return self.time_likelihood(i)*self.backward_priors[i]
    
    def posterior(self,i):
        return self.forward_priors[i].filtered(self.time_likelihood(i)*self.backward_priors[i])
    
    @property
    def forward_posteriors(self):
        res = {}
        for i in self.players:
            res[i] = self.forward_posterior(i)
        return res

    @property
    def backward_posteriors(self):
        res = {}
        for i in self.players:
            res[i] = self.backward_posterior(i)
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
            game = Game(self.within_priors(g),self.results[g],self.games_composition[g])
            for te in range(self.teams(g)):
                names = self.names(g)[te]
                for i in range(len(names)):
                    n = names[i] 
                    max_delta = max(abs(game.likelihood[te][i].mu - self.likelihoods[n][g].mu),max_delta)
                    self.likelihoods[n][g] = game.likelihood[te][i]
        return max_delta
    
    def convergence(self):
        delta = np.inf
        iterations = 0
        while delta > self.epsilon:
            delta = self.iteration()
            print(delta)
            iterations += 1
        return iterations

class History(object):
    def __init__(self
                 , games_composition
                 , results
                 , times=None
                 , time_step=0
                 , prior_dict = {} 
                 , default=Rating(mu=25,sigma=25/3,beta=25/6,noise=25/300)
                 , epsilon=10**-3):
        self.games_composition = map(lambda xs: xs if isinstance(xs[0],list) else [ [x] for x in xs] ,games_composition)
        self.results = results
        self.times = times
        if not self.times is None:
            self.games_composition, self.results, self.times = map(lambda x: list(x),list(zip(*sorted(zip(self.games_composition,self.results, self.times), key=lambda x: x[2]))))
            # = list(zip(*sorted(zip(self.games_composition,self.results, self.times), key=lambda x: x[2])))
            #teams_index_sorted = sorted(zip(self.teams, range(len(self.teams)), self.results), key=lambda x: x[2])
            #teams_sorted , index_sorted, _ = list(zip(*teams_index_sorted )) 
    
            
        #seconds_in_year = 365.25 * 24 * 60 * 60
        if time_step == 'seconds':
            self.time_step = timedelta(seconds=1)
        if time_step == 'minutes':
            self.time_step = timedelta(minutes=1)
        if time_step == 'days':
            self.time_step = timedelta(days=1)
        if time_step == 'weeks':
            self.time_step = timedelta(weeks=1)
        #if time_step = "months":
        #    self.time_step = timedelta(months=1)
        
            
        self.default = default
        self.epsilon = epsilon
                
        self.initial_prior= defaultdict(lambda: self.default)
        self.initial_prior.update(prior_dict)
        
        self.forward_priors = self.initial_prior.copy()
        self.backward_priors = defaultdict(lambda: Gaussian()) 
        
        self.time_line = []
        
        self.learning_curve = {}
    
    def update_index(self,i,t):
        j = i + 1
        while (j < len(self)) and (self.times[j] <= t):
            j += 1
        return j
    
    #def __init__(self,games_composition,results,forward_priors=defaultdict(lambda: Rating()) , time_step=None,epsilon=10**-3):
    def create_time_line_with_time(self):
        i = 0; 
        while i < len(self):
            t = self.times[i]
            j = self.update_index(i,t+self.time_step)
            time = Time(games_composition = self.games_composition[i:j]
                        ,results = self.results[i:j]
                        ,forward_priors = self.forward_priors
                        ,self.times[i:j]
                        )
            self.time_line.append(time)
            i = j 
            
            
            
            
        
        
    def __len__(self):
        return len(self.games_composition)
    
    
    
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
  
def flat(xs):
    if len(xs) == 0 or not isinstance(xs[0],list):
        res = xs
    else:
        res = sum(xs,[])
    return res
      
    
