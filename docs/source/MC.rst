Markov Chain (MC)
=================
A MC is a deterministic model where the transition functions and the generating functions are dependent.
The model first generate an observation and move to the next state according to one unique probability
distributions. More information `here <https://en.wikipedia.org/wiki/Markov_chain>`_. 

Example:

.. image:: pictures/MC.png
   :width: 75 %
   :align: center

.. code-block:: python

   import jajapy as ja
   s0 = ja.MC_state([[0.3,0.3,0.2,0.2],[1,1,2,2],['a','b','a','b']], 0)
   s1 = ja.MC_state([[1.0],['c'],[1]], 1)
   s2 = ja.MC_state([[1.0],['d'],[2]], 2)
   lst_states = [s0, s1, s2]
   model = ja.MC(states=lst_states,initial_state=0,name="My MC")

Model
-----

.. autoclass:: jajapy.MC
   :members:
   :inherited-members:

State
-----

.. autoclass:: jajapy.MC_state
   :members:
   :inherited-members:

Other Functions
---------------

.. autofunction:: jajapy.loadMC

.. autofunction:: jajapy.MC_random

.. autofunction:: jajapy.HMMtoMC