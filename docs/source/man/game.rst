The `Game` class
=================

.. currentmodule:: trueskillthroughtime

We use the `Game` class to model events and perform inference.

.. autoclass:: Game
    :members: posteriors,
              performance
   
Let us return to the example seen on the first page of this manual.

.. code-block::

    >>> a1 = ttt.Player(); a2 = ttt.Player(); a3 = ttt.Player(); a4 = ttt.Player()
    >>> team_a = [ a1, a2 ]
    >>> team_b = [ a3, a4 ]
    >>> g = ttt.Game([team_a, team_b])
    >>> g.teams
    [[Player(Gaussian(mu=0.000, sigma=6.000), beta=1.000, gamma=0.030), Player(Gaussian(mu=0.000, sigma=6.000), beta=1.000, gamma=0.030)], [Player(Gaussian(mu=0.000, sigma=6.000), beta=1.000, gamma=0.030), Player(Gaussian(mu=0.000, sigma=6.000), beta=1.000, gamma=0.030)]]

where the teams' order in the list implicitly defines the game's result: the teams appearing first in the list (lower index) beat those appearing later (higher index). 

Evidence and likelihood
-----------------------

During the initialization, the :code:`Game` class computes the prior prediction of the observed result (the :code:`evidence` atribute) and the approximate likelihood of each player (the :code:`likelihoods` atribute).

.. code-block::

    >>> lhs = g.likelihoods
    >>> round(g.evidence, 3)
    0.5
    

In this case, the evidence is :code:`0.5` because both teams had the same prior skill estimates.

Posterior 
---------

The method :code:`posteriors()` of class :code:`Game` to compute the posteriors.

.. code-block::

    >>> pos = g.posteriors()
    >>> pos[0][0]
    N(mu=2.361, sigma=5.516)

Posteriors can also be found by manually multiplying the likelihoods and priors. 

.. code-block::

    >>> lhs[0][0] * a1.prior
    N(mu=2.361, sigma=5.516)

Team performance
----------------

We can obtain the expected performance of the first team. 

.. code-block::

    >>> g.performance(0)
    N(mu=0.000, sigma=8.602)

Full example
------------

We now analyze a more complex example in which the same four players participate in a multi-team game.
The players are organized into three teams of different sizes: two teams with only one player and the other with two players. 
The result has a single winning team and a tie between the other two losing teams.
Unlike the previous example, we need to use a draw probability greater than zero.

.. code-block::

    >>> ta = [a1]
    >>> tb = [a2, a3]
    >>> tc = [a4]
    >>> teams_3 = [ta, tb, tc]
    >>> result = [1., 0., 0.]
    >>> g = ttt.Game(teams_3, result, p_draw=0.25)
    >>> g.result
    [1.0, 0.0, 0.0]

The team with the highest score is the winner, and the teams with the same score are tied.
In this way, we can specify any outcome including global draws.
The evidence and posteriors can be queried in the same way as before.

.. code-block::

    >>> g.posteriors()
    [[N(mu=3.864, sigma=4.724)], [N(mu=-1.290, sigma=4.776), N(mu=-1.290, sigma=4.776)], [N(mu=-2.574, sigma=4.274)]]
