The `Player` class
===================

.. currentmodule:: trueskillthroughtime

The features of the agents are defined within class :code:`Player`: the prior Gaussian distribution characterized by the mean (:code:`mu`) and the standard deviation (:code:`sigma`), the standard deviation of the performance (:code:`beta`), and the dynamic uncertainty of the skill (:code:`gamma`). 

.. autoclass:: Player
    :members: performances

The default value of :code:`MU = 0.0`, :code:`SIGMA = 6.0`, :code:`BETA = 1.0`, :code:`GAMMA = 0.03`

.. code-block::

    >>> a1 = ttt.Player()
    >>> a1
    Player(Gaussian(mu=0.000, sigma=6.000), beta=1.000, gamma=0.030)
    >>> a2 = ttt.Player(ttt.Gaussian(0.0, 1.0))
    >>> a2
    Player(Gaussian(mu=0.000, sigma=6.000), beta=1.000, gamma=0.030)


We can also create special players who have non-random performances :code:`beta = 0.0`, and whose skills do not change over time :code:`gamma=0.0`.

.. code-block::

    >>> a3 = ttt.Player(beta=0.0, gamma=0.0)
    >>> a3.beta
    0.0
    >>> a3.gamma
    0.0


Performance
------------

The performances :code:`p` are random variables around their unknown true skill :code:`s`,

:math:`p \sim \mathcal{N}(s,\beta^2)`

.. code-block::

    >>> a2.performance()
    N(mu=0.000, sigma=1.414)
    >>> a3.performance()
    N(mu=0.000, sigma=6.000)
