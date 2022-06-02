Multiple Gaussian Observations Hidden Markov Model (MGOHMM)
===========================================================
A GOHMM is similar to a GOHMM, but this time each state will generated several real number according
to different gaussian distributions. Instead of having a unique gaussian ditribution each state has
here the several pairs of parameters *mu* and *sigma*.

.. code-block:: python

   import jajapy as ja
   s0 = ja.MGOHMM_state([[0.9,0.1],[0,1]],[[3.0,5.0],[2.0,4.0]],0)
   s1 = ja.MGOHMM_state([[0.05,0.9,0.04,0.01],[0,1,2,4]],[[0.5,1.5],[2.5,1.5]],1)
   s2 = ja.MGOHMM_state([[0.05,0.8,0.14,0.01],[1,2,3,4]],[[0.2,0.7],[1.0,1.0]],2)
   s3 = ja.MGOHMM_state([[0.05,0.95],[2,3]],[[0.0,0.3],[1.5,5.0]],3)
   s4 = ja.MGOHMM_state([[0.1,0.9],[1,4]],[[2.0,4.0],[0.5,0.5]],4)
   model = ja.MGOHMM([s0,s1,s2,s3,s4],[0.1,0.7,0.0,0.0,0.2],"MGOHMM")

Model
-----

.. autoclass:: jajapy.MGOHMM
   :members:
   :inherited-members:

State
-----

.. autoclass:: jajapy.MGOHMM_state
   :members:
   :inherited-members:

Other Functions
---------------

.. autofunction:: jajapy.loadMGOHMM

.. autofunction:: jajapy.MGOHMM_random
