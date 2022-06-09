Markov Chain (MC)
=================
A MC is a deterministic model where the transition functions and the generating functions are dependent.
The model first generate an observation and move to the next state according to one unique probability
distributions. More information `here <https://en.wikipedia.org/wiki/Markov_chain>`_. 

Example
-------

.. image:: pictures/MC.png
   :width: 75 %
   :align: center

Creation
^^^^^^^^

.. code-block:: python

   import jajapy as ja
   s0 = ja.MC_state([[0.3,0.3,0.2,0.2],[1,1,2,2],['a','b','a','b']], 0)
   s1 = ja.MC_state([[1.0],['c'],[1]], 1)
   s2 = ja.MC_state([[1.0],['d'],[2]], 2)
   lst_states = [s0, s1, s2]
   model = ja.MC(states=lst_states,initial_state=0,name="My MC")

Exploration
^^^^^^^^^^^

.. code-block:: python

	model.tau(0,1,'b') 		 # 0.3
	model.states[0].tau(1,'b')	 # 0.3
	model.observations()	 # ['a','b','c','d']
	model.states[0].observations() # ['a','b']

Running
^^^^^^^

.. code-block:: python

	model.run(5) # returns a list of 5 observations
	s = model.generateSet(10,5) # returns a Set containing 10 traces of size 5
	assert type(s) == Set
	assert sum(s.times) == 10
	assert len(s.sequences[0]) == 5

Analysis
^^^^^^^^

.. code-block:: python

	model.logLikelihood(s) # loglikelihood of this set of traces under this model

Saving/Loading
^^^^^^^^^^^^^^

.. code-block:: python

	model.save("my_mc.txt")
	another_model = ja.loadMC("my_mc.txt")

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