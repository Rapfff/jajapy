Continuous Time Markov Chain (CTMC)
===================================
In a CTMC, each transition from :math:`s` to :math:`s'` generating :math:`\ell` is associated to a exponential probability distribution of parameter :math:`R(s,\ell,s')`.
The probability of this transition to be triggered within :math:`\tau \in \mathbb{R}_{>0}` time-units is :math:`1 - e^{- R(s,\ell,s') \, \tau}`. 
When, from a state :math:`s`, there are more than one outgoing transition, we are in presence of a race condition.
In this case, the first transition to be triggered determines which observation is generated as well as the next state of the CTMC.
According to these dynamics, the time spent in state :math:`s` before any transition occurs, called waiting time, is exponentially distributed with parameter :math:`E(s) = \sum_{\ell \in L}\sum_{s' \in S} R(s,\ell,s')`, called exit-rate of :math:`s`.

In the following picture, the values in the states are the exit-rates. The observations, which are *red, green, blue* are represented by the color of the transitions.

.. image:: pictures/CTMC.png
   :width: 75 %
   :align: center

.. code-block:: python

   import jajapy as ja
   s0 = CTMC_state([[0.3/5,0.5/5,0.2/5],[1,2,3], ['r','g','r']],0)
   s1 = CTMC_state([[0.08,0.25,0.6,0.07],[0,2,2,3], ['r','r','g','b']],1)
   s2 = CTMC_state([[0.5/4,0.2/4,0.3/4],[1,3,3], ['b','g','r']],2)
   s3 = CTMC_state([[0.95/2,0.04/2,0.01/2],[0,0,2], ['r','g','r']],3)
   model = CTMC([s0,s1,s2,s3],0,name="My_CTMC")



Model
-----

.. autoclass:: jajapy.CTMC
   :members:
   :inherited-members:

State
-----

.. autoclass:: jajapy.CTMC_state
   :members:
   :inherited-members:

Other Functions
---------------

.. autofunction:: jajapy.loadCTMC

.. autofunction:: jajapy.CTMC_random

.. autofunction:: jajapy.asynchronousComposition
