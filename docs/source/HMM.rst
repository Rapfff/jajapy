Hidden Markov Model (HMM)
=========================
A HMM is a simple deterministic model where the transition functions and the generating functions are independent.
In other words, the model first generate an observation and then move to the next state according to two independent
probability distributions. More information `here <https://en.wikipedia.org/wiki/Hidden_Markov_model>`_. 

Example
-------

.. image:: pictures/HMM.png
   :width: 75 %
   :align: center

Creation
^^^^^^^^

.. code-block:: python

   import jajapy as ja
   s0 = ja.HMM_state([[0.4,0.6],['x','y']], [[0.5,0.5],[1,2]], 0)
   # same logic for the other states.
   s1 = ja.HMM_state([[0.8,0.2],['a','b']], [[1.0],[3]], 1)
   s2 = ja.HMM_state([[0.1,0.9],['a','b']], [[1.0],[4]], 2)
   s3 = ja.HMM_state([[0.5,0.5],['x','y']], [[0.8,0.1,0.1],[0,1,2]], 3)
   s4 = ja.HMM_state([[1.0],['y']], [[1.0],[3]], 4)
   lst_states = [s0, s1, s2, s3, s4]
   model = ja.HMM(states=lst_states,initial_state=0,name="My HMM")

Exploration
^^^^^^^^^^^

.. code-block:: python

	model.a(0,1) 		 # 0.5
	model.states[0].a(1)	 # 0.5
	model.a(1,3) 		 # 1.0
	model.b(0,'x')		 # 0.4
	model.tau(0,1,'x')	 # 0.2
	model.observations()	 # ['x','y','a','b']
	model.states[0].observations() # ['x','y']

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

	model.save("my_hmm.txt")
	another_model = ja.loadHMM("my_hmm.txt")

Model
-----

.. autoclass:: jajapy.HMM
   :members:
   :inherited-members:

State
-----

.. autoclass:: jajapy.HMM_state
   :members:
   :inherited-members:

Other Functions
---------------

.. autofunction:: jajapy.loadHMM

.. autofunction:: jajapy.HMM_random