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
   from numpy import array
   alphabet=['a','b','c','d']
   nb_states = 3
   s0 = ja.MC_state([(1,'a',0.3),(2,'b',0.3),(1,'a',0.2),(2,'b',0.2)], alphabet, nb_states)
   s1 = ja.MC_state([(1,'c',1.0)], alphabet, nb_states)
   s2 = ja.MC_state([(2,'d',1.0)], alphabet, nb_states)
   lst_states = array([s0, s1, s2])
   model = ja.MC(states=lst_states,alphabet,initial_state=0,name="My MC")
   # print(model)


We can also generate a random MC

.. code-block:: python

	>>> random_model = ja.MC_random(nb_states=3,
					random_initial_state=False,
					alphabet=['a','b','c','d'])

Exploration
^^^^^^^^^^^

.. code-block:: python

	>>> model.tau(0,1,'b') # probability of moving from s0 to s1 seeing 'b' 
	0.3
	>>> model.getAlphabet()	 # all possible observations
	['a','b','c','d']
	>>> model.getAlphabet(0) #all possible observations in state s0
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
	>>> s.times # the first sequence appears three times, the second one four times, etc...
	[3, 4, 2, 1]

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

Other Functions
---------------
.. autofunction:: jajapy.MC_state

.. autofunction:: jajapy.loadMC

.. autofunction:: jajapy.MC_random