Gaussian Observations Hidden Markov Model (GOHMM)
=================================================
A GOHMM is similar to a HMM, but this time the observations are real values generated according
to gaussian distribution. Instead of having a discrete probability ditribution over all the possible
observations, each state has here the two parameters *mu* and *sigma* of a gaussian distribution.

Example:
In the following picture, the dotted arrows represent the initial state probability distribution: we will start
in the yellow state with probability 0.1.

.. image:: pictures/GOHMM.png
   :width: 75 %
   :align: center

.. code-block:: python

   import jajapy as ja
   s0 = ja.GOHMM_state([[0.9,0.1],[0,1]],[3.0,5.0],0)
   s1 = ja.GOHMM_state([[0.05,0.9,0.04,0.01],[0,1,2,4]],[0.5,1.5],1)
   s2 = ja.GOHMM_state([[0.05,0.8,0.14,0.01],[1,2,3,4]],[0.2,0.7],2)
   s3 = ja.GOHMM_state([[0.05,0.95],[2,3]],[0.0,0.3],3)
   s4 = ja.GOHMM_state([[0.1,0.9],[1,4]],[2.0,4.0],4)
   model = ja.GOHMM([s0,s1,s2,s3,s4],[0.1,0.7,0.0,0.0,0.2],name="My GOHMM")


Model
-----

.. autoclass:: jajapy.GOHMM
   :members:
   :inherited-members:

State
-----

.. autoclass:: jajapy.GOHMM_state
   :members:
   :inherited-members:

Other Functions
---------------

.. autofunction:: jajapy.loadGOHMM

.. autofunction:: jajapy.GOHMM_random
