The `Gaussian` class
==========================

.. currentmodule:: trueskillthroughtime

The :code:`Gaussian` class does most of the computation of the packages.

.. autoclass:: Gaussian
    :members: __mul__,
              
    
The default value are :code:`MU = 0.0` and :code:`SIGMA = 6.0`

.. code-block::
    
    >>> N06 = ttt.Gaussian()
    >>> N06
    N(mu=0.000, sigma=6.000)

Others ways to create :code:`Gaussian` objects

.. code-block::

    >>> N01 = ttt.Gaussian(sigma = 1.0)
    >>> N12 = ttt.Gaussian(1.0, 2.0)
    >>> Ninf = ttt.Gaussian(1.0,ttt.inf)
    >>> N01.mu
    0.0
    >>> N01.sigma
    1.0

The class overwrites the addition :code:`+`, subtraction :code:`-`, product :code:`*`, and division :code:`/` to compute the marginal distributions used in the TrueSkill Through Time model.

Product :code:`*`
-----------------

- :math:`\mathcal{N}(x|\mu_1,\sigma_1^2)\mathcal{N}(x|\mu_2,\sigma_2^2) \propto \mathcal{N}(x|\mu_{*},\sigma_{*}^2)`

with :math:`\frac{\mu_{*}}{\sigma_{*}^2} = \frac{\mu_1}{\sigma_1^2} + \frac{\mu_2}{\sigma_2^2}` and :math:`\sigma_{*}^2 = (\frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2})^{-1}`.

.. code-block::
    
    >>> N06 * N12
    N(mu=0.900, sigma=1.897)    
    >>> N12 * Ninf
    N(mu=1.000, sigma=2.000)

    
Division :code:`/`
------------------
 
- :math:`\mathcal{N}(x|\mu_1,\sigma_1^2)/\mathcal{N}(x|\mu_2,\sigma_2^2) \propto \mathcal{N}(x|\mu_{\div},\sigma_{\div}^2)`

with :math:`\frac{\mu_{\div}}{\sigma_{\div}^2} = \frac{\mu_1}{\sigma_1^2} - \frac{\mu_2}{\sigma_2^2}` and :math:`\sigma_{\div}^2 = (\frac{1}{\sigma_1^2} - \frac{1}{\sigma_2^2})^{-1}`.


.. code-block::

    >>> N12 / N06
    N(mu=1.125, sigma=2.121)
    >>> N12 / Ninf
    N(mu=1.000, sigma=2.000)
    
Addition :code:`+`
-------------------

- :math:`\iint \delta(t=x + y) \mathcal{N}(x|\mu_1, \sigma_1^2)\mathcal{N}(y|\mu_2, \sigma_2^2) dxdy =  \mathcal{N}(t|\mu_1+\mu_2,\sigma_1^2 + \sigma_2^2)`

.. code-block::

    >>> N06 + N12
    N(mu=1.000, sigma=6.325)


Substraction :code:`-`
-----------------------


- :math:`\iint \delta(t=x - y) \mathcal{N}(x|\mu_1, \sigma_1^2)\mathcal{N}(y|\mu_2, \sigma_2^2) dxdy =  \mathcal{N}(t|\mu_1-\mu_2,\sigma_1^2 + \sigma_2^2)`

.. code-block::

    >>> N06 - N12
    N(mu=-1.000, sigma=6.325)


Others methods
----------------

.. code-block::

    >>> N06-N12 == ttt.Gaussian(mu=-1.0, sigma=6.324555)
    False
    >>> (N06-N12).isapprox(ttt.Gaussian(mu=-1.0, sigma=6.324555), 1e-6)
    True
    >>> N12.forget(gamma=1, t=1)
    N(mu=1.000, sigma=2.236)

