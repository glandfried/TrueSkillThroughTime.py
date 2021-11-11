Causal model
============

Knowing how individual skills change over time is essential in many areas. 
Since skills are hidden variables, the best we can do is estimating them based on its direct observable consequences: the outcome of problem-solving and competitions. 
Considering only the frequency of positive results as an indicator of the individuals' ability could lead to wrong approximations, mainly because the outcome also depends on the difficulty of the challenge. 
For this reason, all widely used skill estimators are based on pairwise comparisons.
All currently used skill estimators share some variant of the following causal model:

.. image:: ../_static/elo.png

This is a generative model in which skills (:math:`s`) cause the observable results (:math:`r`) mediated by the difference of hidden performances, :math:`d =p_i - p_j`.
Even if the skills are constant at a given point in time, the performances are random variables around their unknown true skill, :math:`p \sim \mathcal{N}(s,\beta^2)`.
The model assumes that the agent with the highest performance wins, :math:`r = (d > 0)`.
Observable variables are painted gray, hidden are transparent, and constants are shown as dots. 


The scale of estimates
----------------------

The standard deviation of performances :math:`\beta`, is the same for all the agents, acts as the scale of the estimates.
A real skill difference of one beta between two agents is equivalent to 76% probability of winning in favor of the stronger agent.
For this reason we choose the default value to be 1.

.. autodata:: BETA


The Prior
---------


The causal model assumes that, at a given time, the skills are constant. However, we do not know this value.
To represent our uncertainty we use a Gaussian distribution.

:math:`p(s) = \mathcal{N}(\mu, \sigma^2)`

- The initial mean (:math:`\mu`) can be freely chosen because it is the difference of skills that matters and not its absolute value.

.. autodata:: MU

- The prior's standard deviation ($\sigma$) must be sufficiently large to include all possible skill hypotheses. For this reason we chose it to be 6 times larger than the standard deviation of the performance.

.. autodata:: SIGMA


The dynamic factor
------------------

Since skills change over time, it is important to incorporate some uncertainty (:math:`\gamma`) after each time step.

:math:`p(s_{t}) = \mathcal{N}(s_{t} | \mu_{{t-1}}, \, \sigma_{{t-1}}^2 + \gamma^2 )`

where :math:`\mu_{t-1}` and :math:`\sigma_{t-1}` are the mean and standard deviation of the skill estimate at the previous time.
As its optimal value is generally relatively low, we chose the default value to be 3% of the standard deviation of the performances.

.. autodata:: GAMMA


The draw probability
--------------------

A rule of thumb states that the probability of a draw must be initialized with the observed frequency of draws.
If in doubt, it is a candidate parameter to be optimized or integrated by the sum rule.
It is used to compute the prior probability of the observed result, so its value may affect an eventual model selection task.
The default value is 0.

.. autodata:: P_DRAW




