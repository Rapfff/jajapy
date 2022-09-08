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
   s0 = ja.MC_state([(1,'a',0.3),(2,'b',0.3),(1,'a',0.2),(2,'b',0.2)], 0)
   s1 = ja.MC_state([(1,'c',1.0)], 1)
   s2 = ja.MC_state([(2,'d',1.0)], 2)
   lst_states = [s0, s1, s2]
   model = ja.MC(states=lst_states,initial_state=0,name="My MC")
   # print(model)


We can also generate a random MC

.. code-block:: python

	>>> random_model = MC_random(nb_states=3,
					random_initial_state=False,
					alphabet=['a','b','c','d'])

Exploration
^^^^^^^^^^^

.. code-block:: python

	>>> model.tau(0,1,'b') 		 
	0.3
	>>> model.states[0].tau(1,'b')	 
	0.3
	>>> model.observations()	 
	['a','b','c','d']
	>>> model.states[0].observations() 
	['a','b']

Running
^^^^^^^

.. code-block:: python

	>>> model.run(5) # returns a list of 5 observations
	['b', 'd', 'd', 'd', 'd']
	>>> s = model.generateSet(10,5) # returns a Set containing 10 traces of size 5
	>>> s.sequences
	[['a', 'c', 'c', 'c', 'c'], ['b', 'd', 'd', 'd', 'd'],
	 ['b', 'c', 'c', 'c', 'c'], ['a', 'd', 'd', 'd', 'd']]
	>>> s.times
	[3, 4, 2, 1]
	>>> s.type
	0

Analysis
^^^^^^^^

.. code-block:: python

	>>> model.logLikelihood(s) # loglikelihood of this set of traces under this model
	-1.4067053583800182

Saving/Loading
^^^^^^^^^^^^^^

.. code-block:: python

	>>> model.save("my_mc.txt")
	>>> another_model = ja.loadMC("my_mc.txt")

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