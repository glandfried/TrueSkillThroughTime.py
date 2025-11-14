# -*- coding: utf-8 -*-
"""
TrueskillThroughTime.py
~~~~~~~~~~~~~~~~~~~~~~~

A Python implementation of the TrueSkill Through Time algorithm for 
estimating time-varying skill levels of players/agents in competitive games.

This package supports:
- Individual and team games
- Multiple observation models (Ordinal, Continuous, Discrete)
- Time-varying skills with drift
- Online and batch learning modes
- Draw probabilities

Main classes:
- Gaussian: Represents a Gaussian distribution with operations
- Player: Represents a player with skill distribution and parameters
- Game: Represents a single game/match with teams and results
- History: Manages a sequence of games and performs inference

:copyright: (c) 2019-2026 by Gustavo Landfried.
:license: BSD, see LICENSE for more details.
"""

import math

inf = math.inf
sqrt2 = math.sqrt(2)
sqrt2pi = math.sqrt(2 * math.pi)
from scipy.stats import norm
from scipy.stats import truncnorm
from collections import defaultdict
import math
from scipy.special import erfcinv as erfcinv_scipy
import scipy
import numpy as np
from scipy.optimize import minimize
import copy


__all__ = ['MU', 'SIGMA', 'Gaussian', 'N01', 'N00', 'Ninf', 'Nms', 'Player', 'Game', 'History' ]

MU = 0.0
SIGMA = 6
PI = SIGMA**-2
TAU = PI * MU

BETA = 1.0  # Performance noise std dev
GAMMA = 0.03  # Skill drift rate

def erfc(x):
    """Complementary error function."""
    return math.erfc(x)

def erfcinv(y):
    """Inverse complementary error function."""
    return erfcinv_scipy(y)

def cdf(x, mu=0, sigma=1):
    """
    Cumulative distribution function of a Gaussian distribution.
    
    Args:
        x: Value at which to evaluate the CDF
        mu: Mean of the distribution (default: 0)
        sigma: Standard deviation (default: 1)
    
    Returns:
        float: Probability that a sample from N(mu, sigma²) is less than or equal to x
    """
    z = -(x - mu) / (sigma * sqrt2)
    return (0.5 * erfc(z))

def pdf(x, mu, sigma):
    """
    Probability density function of a Gaussian distribution.
    
    Args:
        x: Value at which to evaluate the PDF
        mu: Mean of the distribution
        sigma: Standard deviation
    
    Returns:
        float: Probability density at x for N(mu, sigma²)
    """
    normalizer = (sqrt2pi * sigma)**-1
    functional = math.exp( -((x - mu)**2) / (2*sigma**2) )
    return normalizer * functional

def ppf(p, mu, sigma):
    """
    Percent point function (inverse CDF) of a Gaussian distribution.
    
    Args:
        p: Probability value (between 0 and 1)
        mu: Mean of the distribution
        sigma: Standard deviation
    
    Returns:
        float: Value x such that P(X ≤ x) = p for X ~ N(mu, sigma²)
    """
    return mu - sigma * sqrt2  * erfcinv(2 * p)

def v_w(mu, sigma, margin, tie):
    """
    Compute correction factors v and w for truncated Gaussian moments.
    
    These factors are used in the expectation propagation approximation
    for updating beliefs based on win/loss/tie outcomes.
    
    Args:
        mu: Mean of the Gaussian
        sigma: Standard deviation
        margin: Draw margin (determines tie region)
        tie: Boolean indicating if the outcome is a tie
    
    Returns:
        tuple: (v, w) correction factors for mean and variance
    """
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
    return v, w

def trunc(mu, sigma, margin, tie):
    """
    Compute parameters of a truncated Gaussian distribution.
    
    Args:
        mu: Mean of the original Gaussian
        sigma: Standard deviation of the original Gaussian
        margin: Draw margin
        tie: Boolean indicating if the outcome is a tie
    
    Returns:
        tuple: (mu_trunc, sigma_trunc) parameters of the truncated distribution
    """
    v, w = v_w(mu, sigma, margin, tie)
    mu_trunc = mu + sigma * v
    sigma_trunc = sigma * math.sqrt(1-w)
    return mu_trunc, sigma_trunc

def approx(N, margin, tie):
    """
    Create a Gaussian approximation of a truncated distribution.
    
    Args:
        N: Gaussian object representing the prior distribution
        margin: Draw margin
        tie: Boolean indicating if the outcome is a tie
    
    Returns:
        Gaussian: Approximation of the truncated distribution
    """
    mu, sigma = trunc(N.mu, N.sigma, margin, tie)
    return Gaussian(mu, sigma)


def fixed_point_approx(r: int, mu: float, sigma: float, max_iter: int = 16, tol: float = 1e-6) -> tuple[float, float]:
    """
    Gaussian approximation via fixed-point iteration for Poisson observations.
    
    Solves the fixed-point equations for the Gaussian approximation of p(d|r) 
    when the result r is a non-negative score r ~ Poisson(r|exp(d)) and the 
    difference of performance is Gaussian, d ~ N(d|mu, sigma²).
    
    Based on: Guo et al. (2012) "Score-based Bayesian skill learning"
    
    Args:
        r: Observed score difference (non-negative integer)
        mu: Prior mean of performance difference
        sigma: Prior standard deviation of performance difference
        max_iter: Maximum number of iterations (default: 16)
        tol: Convergence tolerance (default: 1e-6)
    
    Returns:
        tuple: (mu_new, sigma_new) parameters of the Gaussian approximation
    """
    sigma2 = sigma**2
    def compute_kappa(k_prev):
        term = k_prev - mu - r * sigma2 - 1
        sqrt_term = math.sqrt(term**2 + 2*sigma2)
        numerator = mu + r*sigma2 - 1 - k_prev + sqrt_term
        return math.log(numerator/(2*sigma2))
    #
    # Initialize kappa
    kappa = 1
    #
    # Fixed-point iteration
    i = 0
    step = inf
    while (i < max_iter) and step > tol:
        kappa_new = compute_kappa(kappa)
        step = abs(kappa_new - kappa)
        kappa = kappa_new
        #print(i, " ", step)
        i += 1
    #
    # Compute final mu_new and sigma2_new
    mu_new = mu + sigma2 * (r - math.exp(kappa))
    sigma2_new = sigma2 / (1 + sigma2 * math.exp(kappa))
    #
    #print(mu_new, sigma2_new)
    return mu_new, math.sqrt(sigma2_new)

def compute_margin(p_draw, sd):
    """
    Compute the draw margin based on draw probability.
    
    Args:
        p_draw: Probability of a draw (between 0 and 1)
        sd: Standard deviation of the performance difference
    
    Returns:
        float: Draw margin (positive value defining the tie region)
    """
    return abs(ppf(0.5-p_draw/2, 0.0, sd ))

class Gaussian(object):
    """
    Represents a Gaussian (normal) distribution with algebraic operations.
    
    This class implements a Gaussian distribution N(mu, sigma²) with support
    for addition, subtraction, multiplication, and division operations that
    correspond to common probabilistic operations.
    
    Attributes:
        mu (float): Mean of the distribution
        sigma (float): Standard deviation of the distribution
    
    Properties:
        pi (float): Precision (inverse variance) = 1/sigma²
        tau (float): Precision-adjusted mean = mu/sigma²
    """

    def __init__(self, mu=MU, sigma=SIGMA):
        """
        Initialize a Gaussian distribution.
        
        Args:
            mu: Mean of the distribution (default: MU=0.0)
            sigma: Standard deviation (default: SIGMA=6)
        
        Raises:
            ValueError: If sigma is negative
        """
        if sigma >= 0.0:
            self.mu, self.sigma = mu, sigma
        else:
            raise ValueError("sigma should be greater than 0")

    def __iter__(self):
        """Allow unpacking: mu, sigma = gaussian_obj"""
        return iter((self.mu, self.sigma))

    def __repr__(self):
        """String representation of the Gaussian."""
        return 'N(mu={:.6f}, sigma={:.6f})'.format(self.mu, self.sigma)

    @property
    def pi(self):
        """Precision: 1/sigma² (returns inf if sigma=0)"""
        if self.sigma > 0.0:
            return 1 / self.sigma**2
        else:
            return inf

    @property
    def tau(self):
        """Precision-adjusted mean: mu/sigma² (returns 0 if sigma=0)"""
        if self.sigma > 0.0:
            return self.mu * self.pi
        else:
            return 0

    def __add__(self, M):
        """
        Addition of independent Gaussians (convolution).
        
        If X ~ N(mu1, sigma1²) and Y ~ N(mu2, sigma2²) are independent,
        then X + Y ~ N(mu1+mu2, sigma1²+sigma2²)
        """
        return Gaussian(self.mu + M.mu, math.sqrt(self.sigma**2 + M.sigma**2))

    def __sub__(self, M):
        """
        Subtraction of independent Gaussians.
        
        If X ~ N(mu1, sigma1²) and Y ~ N(mu2, sigma2²) are independent,
        then X - Y ~ N(mu1-mu2, sigma1²+sigma2²)
        """
        return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 + M.sigma**2))

    def __mul__(self, M):
        """
        Multiplication of Gaussians or scalar multiplication.
        
        - Scalar multiplication: k*N(mu, sigma²) = N(k*mu, (k*sigma)²)
        - Gaussian multiplication (product of PDFs, normalized):
          N(mu1, sigma1²) * N(mu2, sigma2²) ∝ N(mu_new, sigma_new²)
          where precision-based formulas are used for numerical stability
        """
        if (type(M) == float) or (type(M) == int):
            if M == inf:
                return Ninf
            else:
                return Gaussian(M*self.mu, abs(M)*self.sigma)
        if M.pi == 0:
            return self
        if self.pi == 0:
            return M
        _pi = self.pi + M.pi
        return Gaussian((self.tau + M.tau) / _pi, _pi**(-1/2))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, M):
        """
        Division of Gaussians (removes the contribution of M from self).
        
        Used in message passing to compute marginals from joint distributions.
        Computed using precision: pi_new = pi_self - pi_M
        """
        _pi = self.pi - M.pi
        _sigma = inf if _pi == 0.0 else _pi**(-1/2)
        _mu = 0.0 if _pi == 0.0 else (self.tau - M.tau) / _pi
        return Gaussian(_mu, _sigma)

    def delta(self, M):
        """
        Compute the difference between two Gaussians.
        
        Returns:
            tuple: (|mu1 - mu2|, |sigma1 - sigma2|)
        """
        return abs(self.mu - M.mu), abs(self.sigma - M.sigma)

    def cdf(self, x):
        """Evaluate CDF at x using scipy.stats.norm"""
        return norm(*self).cdf(x)

    def isapprox(self, M, tol=1e-5):
        """
        Check if two Gaussians are approximately equal.
        
        Args:
            M: Another Gaussian object
            tol: Tolerance for comparison (default: 1e-5)
        
        Returns:
            bool: True if both mu and sigma are within tolerance
        """
        return (abs(self.mu - M.mu) < tol) and (abs(self.sigma - M.sigma) < tol)


def suma(Ns):
    """
    Sum a list of independent Gaussian random variables.
    
    Args:
        Ns: List of Gaussian objects
    
    Returns:
        Gaussian: Sum of all Gaussians in the list
    """
    res = Gaussian(0, 0)  # Identity element for sum
    for N in Ns:
        res = res + N
    return res


def producto(Ns):
    """
    Product of Gaussian PDFs (normalized).
    
    Args:
        Ns: List of Gaussian objects
    
    Returns:
        Gaussian: Product of all Gaussians in the list
    """
    res = Gaussian(0, inf)  # Identity element for product
    for N in Ns:
        res = res * N
    return res



N01 = Gaussian(0, 1)
N00 = Gaussian(0, 0)
Ninf = Gaussian(0, inf)
Nms = Gaussian(MU, SIGMA)

class Player(object):
    """
    Represents a player/agent with skill distribution and performance parameters.
    
    Attributes:
        prior (Gaussian): Player's skill distribution
        beta (float): Standard deviation of performance noise (within-game variability)
        gamma (float): Skill drift rate per time unit (between-game learning/decay)
    """

    def __init__(self, prior=Gaussian(MU, SIGMA), beta=BETA, gamma=GAMMA, prior_draw=Ninf):
        """
        Initialize a Player.
        
        Args:
            prior: Gaussian representing the player's skill (default: N(0, 6))
            beta: Performance noise std dev (default: 1.0)
            gamma: Skill drift rate (default: 0.03)
            prior_draw: Reserved for future use
        """
        self.prior = prior
        self.beta = beta
        self.gamma = gamma
    def performance(self):
        """
        Generate a performance distribution: skill + noise.
        
        Returns:
            Gaussian: Performance ~ N(skill_mu, skill_sigma² + beta²)
        """
        return self.prior + Gaussian(0, self.beta)

    def __repr__(self):
        """String representation of the Player."""
        return 'Player(Gaussian(mu=%.3f, sigma=%.3f), beta=%.3f, gamma=%.3f)' % (
            self.prior.mu, self.prior.sigma, self.beta, self.gamma)


class team_variable(object):
    """
    Internal class representing a team's performance variable in message passing.
    
    Attributes:
        prior: Team performance prior distribution
        likelihood_lose: Likelihood message from losing constraint
        likelihood_win: Likelihood message from winning constraint
    """

    def __init__(self, prior, likelihood_lose=Ninf, likelihood_win=Ninf):
        """Initialize team variable with prior and likelihood messages."""
        self.prior = prior
        self.likelihood_lose = likelihood_lose
        self.likelihood_win = likelihood_win

    def __repr__(self):
        """String representation of the team variable."""
        return f'Team(prior={self.prior}, likelihood_lose={self.likelihood_lose}, likelihood_win={self.likelihood_win}'

    @property
    def p(self):
        """Full posterior: prior * all likelihoods"""
        return self.prior * self.likelihood_lose * self.likelihood_win

    @property
    def posterior_win(self):
        """Posterior after incorporating lose constraint"""
        return self.prior * self.likelihood_lose

    @property
    def posterior_lose(self):
        """Posterior after incorporating win constraint"""
        return self.prior * self.likelihood_win

    @property
    def likelihood(self):
        """Combined likelihood from both constraints"""
        return self.likelihood_win * self.likelihood_lose


class diff_messages(object):
    """
    Internal class for performance difference variables in message passing.
    
    Attributes:
        prior: Prior distribution of performance difference
        likelihood: Likelihood message from outcome observation
    """

    def __init__(self, prior, likelihood=Ninf):
        """Initialize difference message with prior and likelihood."""
        self.prior = prior
        self.likelihood = likelihood

    @property
    def p(self):
        """Posterior: prior * likelihood"""
        return self.prior * self.likelihood


class Game(object):
    """
    Represents a single game/match with multiple teams and an outcome.
    
    This class performs Bayesian inference on player skills given a game result
    using expectation propagation with Gaussian message passing.
    
    Attributes:
        teams: List of teams, where each team is a list of Player objects
        result: Outcome for each team (lower is better, ties have equal values)
        p_draw: Probability of a draw (defines draw margin)
        weights: Contribution weight of each player in each team
        obs: Observation model - "Ordinal" (ranking), "Continuous" (score), or "Discrete" (count)
        evidence: Marginal likelihood of the observed outcome
        likelihoods: Likelihood messages for each player's skill
    
    Example:
        >>> playerA = Player()
        >>> playerB = Player()
        >>> playerC = Player()
        >>> teams = [[playerA, playerB], [playerC]]
        >>> result = [1, 2]  # First team wins
        >>> game = Game(teams, result)
        >>> posteriors = game.posteriors()
    """

    def __init__(self, teams, result=[], p_draw=0.0, weights=[], obs="Ordinal"):
        """
        Initialize a Game.
        
        Args:
            teams: List of teams, each team is a list of Player objects
            result: List of outcomes (one per team). Lower is better. 
                   Equal values indicate ties. Default: decreasing order
            p_draw: Draw probability, between 0 and 1 (default: 0.0)
            weights: Player contribution weights (default: all 1.0)
            obs: Observation model - "Ordinal", "Continuous", or "Discrete" (default: "Ordinal")
        """
        g = self
        g.teams = teams
        g.result = result if len(result)==len(teams) else list(range(len(teams)-1,-1,-1))
        if not weights:
            weights = [[1.0 for p in t] for t in teams]
        g.weights = weights
        g.p_draw = p_draw
        g.o = g.orden()
        g.t = g.performance_teams()
        g.d = g.difference_teams()
        if obs=="Ordinal":
            g.tie = [g.result[g.o[e]]==g.result[g.o[e+1]] for e in range(len(g.d))]
            g.margin = g.margin()
        else:
            g.tie = None
            g.margin = None
        g.obs = obs
        g.evidence = 1.0
        g.likelihoods = self.likelihoods()

    def __repr__(self):
        """String representation of the Game."""
        return f'{self.teams}'

    def __len__(self):
        """Return number of teams in the game."""
        return len(self.teams)

    def orden(self):
        """
        Compute the ordering of teams by result (best to worst).
        
        Returns:
            list: Indices of teams sorted by result (descending order)
        """
        return [i[0] for i in sorted(enumerate(self.result), key=lambda x: x[1], reverse=True)]

    def margin(self):
        g = self
        res = []
        for e in range(len(g.d)):
            sd = math.sqrt(\
                sum([a.beta**2 for a in g.teams[g.o[e]]]) +\
                sum([a.beta**2 for a in g.teams[g.o[e+1]]]))
            compute_margin(g.p_draw, sd)
            res.append(0.0 if g.p_draw == 0.0 else compute_margin(g.p_draw, sd))
        return res

    def performance_individuals(self):
        # Generate individual performances by adding noise to skills
        res = []
        for t in range(len(self.teams)):
            res.append([])  # Team container
            for i in range(len(self.teams[t])):
                res[-1].append(self.teams[t][i].performance() * (self.weights[t][i]))
        return res

    def performance_teams(self):
        # Sum of individual performances
        res = []
        for team in self.performance_individuals():
            res.append(team_variable(suma(team)))
        return res

    def difference_teams(self):
        g = self
        res = []
        for e in range(len(g) - 1):
            res.append(diff_messages(g.t[g.o[e]].prior - g.t[g.o[e + 1]].prior))
        return res

    def partial_evidence(self, i_d):
        """Compute partial evidence for a difference variable."""
        g = self
        mu, sigma = g.d[i_d].prior
        if self.obs == "Ordinal":
            if g.tie[i_d]:
                self.evidence *= cdf(g.margin[i_d], mu, sigma) - cdf(-g.margin[i_d], mu, sigma)
            else:
                self.evidence *= 1 - cdf(g.margin[i_d], mu, sigma)
        elif self.obs == "Continuous":
            self.evidence *= pdf(self.result[g.o[i_d]] - self.result[g.o[i_d + 1]], mu, sigma)
        elif self.obs == "Discrete":
            r = self.result[g.o[i_d]]-self.result[g.o[i_d+1]]
            # Monte Carlo Solution
            N = 5000
            hardcoded_lower_bound = 1/(2*N)
            evidence = sum(r == scipy.stats.poisson.rvs(mu=np.exp(scipy.stats.norm.rvs(size=N,loc=mu,scale=sigma))))/N
            self.evidence *= hardcoded_lower_bound + evidence
            #
            # Version Guo et al:
            #r=3
            #lmbda_i = math.exp(3)
            #lmbda_j = math.exp(0)
            #evidence = sum([math.exp(-(lmbda) * (lmbda)**(k/2) * np.i0(2*math.sqrt(lmbda)) for k in range(1,101)])

    #
    def likelihood_difference(self, i_d):
        g = self
        if g.obs == "Ordinal":
            return approx(g.d[i_d].prior, g.margin[i_d], g.tie[i_d]) / g.d[i_d].prior
        elif g.obs == "Continuous":
            return Gaussian(self.result[g.o[i_d]] - self.result[g.o[i_d + 1]], 0.0)
        elif g.obs == "Discrete":
            r = self.result[g.o[i_d]] - self.result[g.o[i_d + 1]]
            mu, sigma = g.d[i_d].prior
            return Gaussian(*fixed_point_approx(r, mu, sigma)) / g.d[i_d].prior

    def likelihood_convergence(self):
        """Iterate likelihood messages until convergence."""
        g = self
        for i in range(5):  # Convergence iterations
            for e in range(len(g.d) - 1):
                g.d[e].prior = g.t[g.o[e]].posterior_win - g.t[g.o[e + 1]].posterior_lose
                if i == 0:
                    g.partial_evidence(e)
                g.d[e].likelihood = g.likelihood_difference(e)
                likelihood_lose = g.t[g.o[e]].posterior_win - g.d[e].likelihood
                g.t[g.o[e + 1]].likelihood_lose = likelihood_lose
            for e in range(len(g.d) - 1, 0, -1):
                g.d[e].prior = g.t[g.o[e]].posterior_win - g.t[g.o[e + 1]].posterior_lose
                if i == 0 and e == len(g.d) - 1:
                    g.partial_evidence(e)
                g.d[e].likelihood = g.likelihood_difference(e)
                likelihood_win = g.t[g.o[e + 1]].posterior_lose + g.d[e].likelihood
                g.t[g.o[e]].likelihood_win = likelihood_win
        if len(g.d) == 1:
            g.partial_evidence(0)
            g.d[0].prior = g.t[g.o[0]].posterior_win - g.t[g.o[1]].posterior_lose
            g.d[0].likelihood = g.likelihood_difference(0)
        g.t[g.o[0]].likelihood_win = g.t[g.o[1]].posterior_lose + g.d[0].likelihood
        g.t[g.o[-1]].likelihood_lose = g.t[g.o[-2]].posterior_win - g.d[-1].likelihood
        return [g.t[e].likelihood for e in range(len(g.t))]

    def likelihood_performance(self):
        """Compute likelihood messages for individual performances."""
        g = self
        performance_individuals = g.performance_individuals()
        likelihood_teams = g.likelihood_convergence()
        # te = p1 + p2 + p3  <=>  p1 = te - (p2 + p3) = te - te_without_i
        res = []
        for e in range(len(g.teams)):
            res.append([])
            for i in range(len(g.teams[e])):
                te = likelihood_teams[e]
                te_without_i = Gaussian(
                    g.t[e].prior.mu - performance_individuals[e][i].mu,
                    math.sqrt(g.t[e].prior.sigma**2 - performance_individuals[e][i].sigma**2))
                w_i = g.weights[e][i]
                inv_w_i = inf if w_i == 0 else 1 / w_i
                res[-1].append(inv_w_i * (te - te_without_i))
        return res

    def likelihood_skill(self):
        """Compute likelihood messages for player skills."""
        res = []
        lh_p = self.likelihood_performance()
        for e in range(len(lh_p)):
            res.append([])
            for i in range(len(lh_p[e])):
                noise = Gaussian(0, self.teams[e][i].beta)
                res[-1].append(lh_p[e][i] - noise)
        return res

    def likelihood_analytic(self):
        """Compute likelihoods analytically for 2-team games."""
        g = self
        g.partial_evidence(0)
        psi, vartheta = g.d[0].prior
        psi_div, vartheta_div = g.likelihood_difference(0)
        res = []
        for t in range(len(g.t)):
            res.append([])
            lose_case = (t == 1)
            for i in range(len(g.teams[g.o[t]])):
                mu_i, sigma_i = g.teams[g.o[t]][i].prior
                w_i = g.weights[g.o[t]][i]
                inv_w_i = inf if w_i == 0 else 1 / w_i
                mu_analytic = mu_i + inv_w_i * (-psi + psi_div) * (-1)**(lose_case)
                sigma_analytic = math.sqrt(
                    (inv_w_i**2) * vartheta_div**2 + (inv_w_i**2) * vartheta**2 - sigma_i**2)
                res[-1].append(Gaussian(mu_analytic, sigma_analytic))
        return [res[0], res[1]] if g.o[0] < g.o[1] else [res[1], res[0]]

    def likelihoods(self):
        """Compute likelihood messages using appropriate method."""
        if len(self.teams) == 2:
            return self.likelihood_analytic()
        else:
            return self.likelihood_skill()

    def posteriors(self):
        """
        Compute posterior skill distributions for all players.
        
        Combines each player's prior skill with the likelihood from the game outcome.
        
        Returns:
            list: Nested list of Gaussian posteriors, structured as [team][player]
        """
        g = self
        res = []
        for e in range(len(g.teams)):
            res.append([])
            for i in range(len(g.teams[e])):
                res[-1].append(g.teams[e][i].prior * g.likelihoods[e][i])
        return res


class Skill(object):
    """
    Internal class representing a player's skill at a specific time point.
    
    Stores messages from forward/backward passes and likelihoods from games.
    Used internally by History for temporal inference.
    
    Attributes:
        bevents: Indices of games (within batch) this player participated in
        forward: Forward message from past games
        backward: Backward message from future games
        likelihoods: List of likelihood messages from games at this time
        online: Online estimate (for online learning mode)
    """
    def __init__(self, bevents=[], forward=Ninf, backward=Ninf, likelihoods=[]):
        """Initialize a Skill variable."""
        self.bevents = bevents
        self.forward = forward
        self.backward = backward
        self.likelihoods = likelihoods
        self.online = None

    def __repr__(self):
        """String representation of the Skill."""
        return f'Skill(events={self.bevents})'

    @property
    def posterior(self):
        """Full posterior: forward * backward * all likelihoods"""
        return self.forward * self.backward * producto(self.likelihoods)

    @property
    def forward_posterior(self):
        """Forward posterior: forward message * likelihoods"""
        return self.forward * producto(self.likelihoods)

    @property
    def backward_posterior(self):
        """Backward posterior: backward message * likelihoods"""
        return self.backward * producto(self.likelihoods)

    def likelihood(self, e):
        """Get likelihood for event e."""
        i = self.bevents.index(e)
        return self.likelihoods[i]

    def update_likelihood(self, e, likelihood):
        """Update likelihood for event e and return step size."""
        i = self.bevents.index(e)
        step = likelihood.delta(self.likelihoods[i])
        self.likelihoods[i] = likelihood
        return step

    def serialize(self):
        """Serialize skill object to dictionary."""
        return {
            'bevents': self.bevents,
            'forward': (self.forward.mu, self.forward.sigma),
            'backward': (self.backward.mu, self.backward.sigma),
            'likelihoods': [(l.mu, l.sigma) for l in self.likelihoods],
            'online': None if not self.online else (self.online.mu, self.online.sigma)
        }

from enum import Enum


class GameType(Enum):
    """
    Enumeration of observation models for game outcomes.
    
    - Ordinal: Ranking/placement (win/loss/draw)
    - Continuous: Continuous scores (e.g., time, distance)
    - Discrete: Discrete counts (e.g., goals, points scored)
    """
    Ordinal = 0
    Continuous = 1
    Discrete = 2


class History(object):
    """
    Manages a sequence of games and performs temporal Bayesian skill inference.
    
    This is the main class for TrueSkill Through Time. It maintains a history
    of games and computes time-varying skill estimates for all players using
    Gaussian message passing with forward and backward iterations.
    
    Attributes:
        size: Total number of games
        batches: List of game compositions grouped by time
        bresults: List of game results grouped by time
        btimes: List of time points
        bskills: Skill variables for each player at each time
        mu: Default prior mean for new players
        sigma: Default prior std dev for new players
        beta: Performance noise std dev
        gamma: Skill drift rate per time unit
        p_draw: Draw probability
        priors: Dictionary mapping player names to Player objects
        online: If True, use online learning mode
    
    Example:
        >>> composition = [[["a"], ["b"]], [["b"], ["c"]], [["c"], ["a"]]]
        >>> results = [[1, 2], [1, 2], [1, 2]]
        >>> history = History(composition, results)
        >>> history.convergence()
        >>> learning_curves = history.learning_curves()
    """

    def __init__(self, composition, results=[], times=[], priors=None, mu=0, sigma=3, beta=1, 
                 gamma=0.15, p_draw=0.0, online=False, weights=[], obs=[]):
        """
        Initialize a History of games.
        
        Args:
            composition: List of games, where each game is a list of teams,
                        and each team is a list of player names (strings)
            results: List of outcomes for each game (default: empty, uses rank order)
            times: Time point for each game (default: sequential integers)
            priors: Dict mapping player names to Player objects (default: None)
            mu: Default prior mean for new players (default: 0)
            sigma: Default prior std dev for new players (default: 3)
            beta: Performance noise std dev (default: 1)
            gamma: Skill drift rate per time unit (default: 0.15)
            p_draw: Draw probability (default: 0.0)
            online: Enable online learning mode (default: False)
            weights: Player contribution weights for each game (default: all 1.0)
            obs: Observation model for each game - "Ordinal", "Continuous", or "Discrete" (default: all "Ordinal")
        
        Raises:
            ValueError: If input dimensions are inconsistent or parameters are invalid
        """
        self.check_input(composition, results, times, priors, mu, sigma, beta, gamma, p_draw, weights, obs)
        
        self.size = 0
        self.batches = []; self.bresults = []; self.btimes = []
        self.bskills = []; self.bweights = []; self.bobs = []
        self.bevidence = []
        self.init_batches(composition, results, times, weights, obs)
        self.mu = mu; self.sigma = sigma
        self.beta = beta; self.gamma = gamma
        self.p_draw = p_draw
        self.priors = defaultdict(
            lambda: Player(Gaussian(mu, sigma), beta, gamma), 
            priors if priors else {})
        self._last_message = None
        self._last_time = None
        self.online = online
        self.b_until = 0 if online else len(self.batches)

    def __repr__(self):
        """String representation of the History."""
        return f'History(Events={self.size})'

    def check_input(self, composition, results, times, priors, mu, sigma, beta, gamma, p_draw, weights, obs):
        """Validate input parameters."""
        self.check_data(composition, results, times, priors, weights, obs)
        if sigma < 0.0: raise ValueError("sigma < 0.0")
        if beta < 0.0: raise ValueError("beta < 0.0")
        if gamma < 0.0: raise ValueError("gamma < 0.0")
        if p_draw < 0.0 or p_draw > 1.0 : raise ValueError("p_draw < 0.0 or p_draw > 1.0")
    #
    def check_data(self, composition, results, times, priors, weights, obs):
        if results and (len(composition) != len(results)): raise ValueError("len(composition) != len(results)")
        if times and (len(composition) != len(times)): raise ValueError("len(composition) != len(times)")
        if (not priors is None) and (not isinstance(priors, dict)): raise ValueError("not isinstance(priors, dict)")
        if weights and (len(composition) != len(weights)): raise ValueError("len(composition) != len(weights)")
        if obs and (len(composition) != len(obs)): raise ValueError("len(composition) != len(obs)")
    #
    def init_batches(self, composition, results, times, weights, obs):
        """Initialize batches from composition data."""
        times = list(range(len(composition))) if not times else times
        last_time = -inf
        i_b = -1  # Always add to last batch
        for i_e in orden(times, reverse=False):
            self.size += 1
            t = times[i_e]
            if t != last_time:
                e = 0
                self.btimes.append(t); self.batches.append([])
                self.bresults.append([]); self.bweights.append([])
                self.bskills.append({}); self.bobs.append([])
                self.bevidence.append([])
                last_time = t
            self.add_to_batch(i_b, i_e, composition, results, times, weights, obs)

    def add_history(self, composition, results=[], times=[], priors=None, weights=[], obs=[]):
        """
        Add new games to an existing History.
        
        Args:
            composition: List of games to add
            results: List of outcomes for new games (default: empty)
            times: Time points for new games (default: sequential from last time)
            priors: Dict of Player objects for new players (default: None)
            weights: Player contribution weights (default: all 1.0)
            obs: Observation models (default: all "Ordinal")
        """
        self.check_data(composition, results, times, priors, weights, obs)
        
        # Merge priors dict (keeping first definition)
        self.priors = (priors if priors else {}) | self.priors
        
        last_t = self.btimes[-1] if self.btimes else 0
        times = list(range(last_t, len(composition) + last_t)) if not times else times
        for i in range(len(composition)):
            t = times[i]
            if t in self.batches:
                i_b = self.batches.index(times[i])
            else:
                i_b = len(self.batches)
                self.btimes.append(t); self.batches.append([])
                self.bresults.append([]); self.bweights.append([])
                self.bskills.append({}); self.bobs.append([])
                self.bevidence.append([])
            self.add_to_batch(i_b, i, composition, results, times, weights, obs)

    def add_to_batch(self, i_b, i, composition, results, times, weights, obs):
        """Add a single game to a batch."""
        self.batches[i_b].append(composition[i])
        self.bresults[i_b].append(results[i] if results else [])
        self.bweights[i_b].append(weights[i] if weights else [])
        self.bobs[i_b].append(GameType[obs[i]].value if obs else GameType["Ordinal"].value)
        self.bevidence[i_b].append(None)
        
        e = len(self.batches[i_b]) - 1
        for team in composition[i]:
            for name in team:
                if name in self.bskills[i_b]:
                    self.bskills[i_b][name].bevents.append(e)
                    self.bskills[i_b][name].likelihoods.append(Ninf)
                else:
                    self.bskills[i_b][name] = Skill(bevents=[e], likelihoods=[Ninf])

    def _in_skills(self, b, forward):
        """Propagate messages into a batch (with skill drift)."""
        h = self
        for name in h.bskills[b]:
            old_t = h._last_time[name]
            elapsed = abs(h.btimes[b] - old_t) if (old_t is not None) else 0
            gamma = h.priors[name].gamma
            receive = h._last_message[name] + Gaussian(
                0, min(math.sqrt(elapsed * (gamma**2)), 1.67 * self.sigma))
            if forward:
                h.bskills[b][name].forward = receive
                if h.online and not h.bskills[b][name].online:
                    h.bskills[b][name].online = receive
            else:
                h.bskills[b][name].backward = receive

    def _up_skills(self, b, e):
        """Update skills based on a single game within a batch."""
        h = self
        g = Game(h.within_priors(b, e), h.bresults[b][e], h.p_draw, h.bweights[b][e], 
                 obs=GameType(self.bobs[b][e]).name)
        likelihoods = g.likelihoods
        if self.online and not self.bevidence[b][e]:
            self.bevidence[b][e] = g.evidence
        if not self.online:
            self.bevidence[b][e] = g.evidence
        mu_step_max, sigma_step_max = (0, 0)
        for t in range(len(h.batches[b][e])):
            for i in range(len(h.batches[b][e][t])):
                name = h.batches[b][e][t][i]
                mu_step, sigma_step = h.bskills[b][name].update_likelihood(e, likelihoods[t][i])
                mu_step_max = max(mu_step_max, mu_step)
                sigma_step_max = max(sigma_step_max, sigma_step)
        return (mu_step_max, sigma_step_max)

    def _out_skills(self, b, forward):
        """Propagate messages out of a batch."""
        h = self
        for name in h.bskills[b]:
            h._last_time[name] = h.btimes[b]
            if forward:
                h._last_message[name] = h.bskills[b][name].forward_posterior
            else:
                h._last_message[name] = h.bskills[b][name].backward_posterior

    def within_priors(self, b, e, online=False):
        """Get player priors for a game (excluding its own likelihood)."""
        h = self
        priors = []
        for t in range(len(h.batches[b][e])):
            priors.append([])
            for i in range(len(h.batches[b][e][t])):
                name = h.batches[b][e][t][i]
                if not online:
                    prior = h.bskills[b][name].posterior / h.bskills[b][name].likelihood(e)
                else:
                    prior = h.bskills[b][name].online
                priors[-1].append(Player(prior, h.priors[name].beta, h.priors[name].gamma))
        return priors

    def forward_propagation(self):
        """
        Forward pass: propagate skill estimates from past to future.
        
        Processes games chronologically, updating beliefs about player skills
        as we move forward in time.
        
        Returns:
            tuple: (max_mu_step, max_sigma_step) - maximum likelihood change 
        """
        h = self
        h._last_message = defaultdict(
            lambda: Gaussian(h.mu, h.sigma), 
            {k: v.prior for k, v in h.priors.items()})
        h._last_time = defaultdict(lambda: None)
        mu_step_max = 0; sigma_step_max = 0;
        for b in range(h.b_until):
            h._in_skills(b, forward=True)
            for e in range(len(h.batches[b])):
                mu_step, sigma_step = h._up_skills(b, e)
                mu_step_max = max(mu_step_max, mu_step); sigma_step_max = max(sigma_step_max, sigma_step)
            h._out_skills(b, forward=True)
        return (mu_step_max, sigma_step_max)

    def backward_propagation(self):
        """
        Backward pass: propagate skill estimates from future to past.
        
        Processes games in reverse chronological order, updating beliefs
        about player skills based on future information.
        
        Returns:
            tuple: (max_mu_step, max_sigma_step) - maximum likelihood change
        """
        h = self
        h._last_message = defaultdict(lambda: Gaussian(0.0, math.inf))
        h._last_time = defaultdict(lambda: None)
        mu_step_max = 0; sigma_step_max = 0;
        h._out_skills(h.b_until-1, forward=False)
        for b in reversed(range(h.b_until-1)):
            h._in_skills(b, forward=False)
            for e in range(len(h.batches[b])):
                mu_step, sigma_step = h._up_skills(b, e)
                mu_step_max = max(mu_step_max, mu_step); sigma_step_max = max(sigma_step_max, sigma_step)
            h._out_skills(b, forward=False)
        h._last_message = None
        h._last_time = None
        return (mu_step_max, sigma_step_max)

    def iteration(self):
        """
        Perform one complete iteration: forward pass followed by backward pass.
        
        Returns:
            tuple: (max_mu_step, max_sigma_step) - maximum likelihood change across both passes
        """
        mu_forward, sigma_forward = self.forward_propagation()
        mu_backward, sigma_backward = self.backward_propagation()
        return (max(mu_forward, mu_backward), max(sigma_forward, sigma_backward))

    def convergence(self, iterations=8, epsilon=0.00001, verbose=True):
        """
        Run iterative message passing until convergence.
        
        Performs forward and backward passes, iterating until convergence
        or maximum iterations reached. In online mode, processes batches
        sequentially.
        
        Args:
            iterations: Maximum number of iterations per batch (default: 8)
            epsilon: Convergence threshold for message updates (default: 1e-5)
            verbose: Print iteration progress (default: True)
        
        Returns:
            tuple: (final_step, num_iterations) - convergence metrics
        """
        i = 0
        delta = math.inf
        self.unveil_batch()
        for _ in range(1+len(self.batches)-self.b_until):
            i = 0; delta = math.inf
            if verbose: print("Batch = ", self.b_until)
            while i < iterations and delta > epsilon:
                if verbose: print("Iteration = ", i, end=" ")
                step = self.iteration(); delta = max(step)
                i += 1
                if verbose: print(", step = ", step)
            self.unveil_batch()
        return step, i

    def unveil_batch(self):
        """Unveil the next batch for online processing."""
        if self.b_until < len(self.batches):
            self.b_until += 1

    def learning_curves(self, who=None, online=False):
        """
        Extract learning curves (skill over time) for players.
        
        Args:
            who: List of player names to extract (default: None, returns all)
            online: Use online estimates instead of batch posteriors (default: False)
        
        Returns:
            dict: Maps player names to lists of (time, skill) tuples
        
        Example:
            >>> lc = history.learning_curves(who=["alice", "bob"])
            >>> for time, skill in lc["alice"]:
            ...     print(f"Time {time}: mu={skill.mu:.2f}, sigma={skill.sigma:.2f}")
        """
        h = self
        res = dict()
        for b in range(len(h.bskills)):
            time = h.btimes[b]
            for name in h.bskills[b]:
                if (who is None) or (name in who):
                    if self.online and online:
                        skill = h.bskills[b][name].online
                    else:
                        skill = h.bskills[b][name].posterior
                    t_p = (time, skill)
                    if name in res:
                        res[name].append(t_p)
                    else:
                        res[name] = [t_p]
        return res

    def log_evidence(self):
        """
        Compute the log marginal likelihood of all observed game outcomes.
        
        This is the log probability of the observed results given the model.
        Useful for model comparison and hyperparameter tuning.
        
        Returns:
            float: Sum of log evidences across all games
        """
        return sum([math.log(evidence) for bevidence in self.bevidence 
                    for evidence in bevidence if evidence])

    def geometric_mean(self):
        """
        Compute the geometric mean of game outcome probabilities.
        
        Returns:
            float: exp(log_evidence / num_games) - average probability per game
        """
        unveil_size = sum([not (e is None) for bevidence in self.bevidence 
                          for e in bevidence])
        return math.exp(self.log_evidence() / unveil_size)

    def __getstate__(self):
        """
        Prepare History for pickling (serialization).
        
        Returns:
            dict: Serializable state dictionary
        """
        state = copy.deepcopy(self.__dict__.copy())

        # Convert defaultdict to regular dict
        state['priors'] = dict(self.priors)

        # Remove unpickleable lambda function
        del state['_last_message']
        del state['_last_time']

        # Convert all Skill objects to their serializable form
        for bskill in state['bskills']:
            for name in bskill:
                bskill[name] = bskill[name].serialize()
        return state

    def __setstate__(self, state):
        """
        Restore History from pickled state (deserialization).
        
        Args:
            state: Serialized state dictionary
        """
        # Restore defaultdict with proper default factory
        priors_dict = state['priors']
        state['priors'] = defaultdict(
            lambda: Player(Gaussian(state['mu'], state['sigma']),
                         state['beta'],
                         state['gamma']),
            priors_dict
        )

        # Restore Skill objects
        for bskill in state['bskills']:
            for name in bskill:
                skill_dict = bskill[name]
                bskill[name] = Skill(
                    bevents=skill_dict['bevents'],
                    forward=Gaussian(*skill_dict['forward']),
                    backward=Gaussian(*skill_dict['backward']),
                    likelihoods=[Gaussian(*l) for l in skill_dict['likelihoods']]
                )
                if skill_dict['online'] is not None:
                    bskill[name].online = Gaussian(*skill_dict['online'])

        # Restore None values for unpickleable attributes
        state['_last_message'] = None
        state['_last_time'] = None

        self.__dict__.update(state)

def orden(xs, reverse=True):
    """
    Return indices that sort a list.
    
    Args:
        xs: List to sort
        reverse: Sort in descending order (default: True)
    
    Returns:
        list: Indices of sorted elements
    """
    return [i[0] for i in sorted(enumerate(xs), key=lambda x: x[1], reverse=reverse)]






