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

	>>> import jajapy as ja
	>>> transitions = [(0,1,0.5),(0,2,0.5),(1,3,1.0),(2,4,1.0),
	>>> 			   (3,0,0.8),(3,1,0.1),(3,2,0.1),(4,3,1.0)]
	>>> emission = [(0,"x",0.4),(0,"y",0.6),(1,"a",0.8),(1,"b",0.2),
	>>> 			(2,"a",0.1),(2,"b",0.9),(3,"x",0.5),(3,"y",0.5),(4,"y",1.0)]
	>>> original_model = ja.createHMM(transitions,emission,initial_state=0,name="My HMM")

We can also generate a random HMM

.. code-block:: python

	>>> random_model = ja.HMM_random(number_states=5,
					random_initial_state=False,
					alphabet=['x','y','a','b'])

Exploration
^^^^^^^^^^^

.. code-block:: python

	>>> model.a(0,1) #probability of going from s0 to s1
	0.5
	>>> model.a(1,3) #probability of going from s1 to s3
	1.0
	>>> model.b(0,'x') #probability of seeing 'x' while in s0
	0.4
	>>> model.tau(0,1,'x') #probability of going from s0 to s1 seeing 'x'
	0.2
	>>> model.getAlphabet() #all possible observations
	['x','y','a','b']
	>>> model.getAlphabet(0) #all possible observations in s0
	['x','y']

Running
^^^^^^^

.. code-block:: python

	>>> model.run(5) # Generate a run of length 5, i.e. returns a list of 5 observations
	['y', 'a', 'y', 'a', 'y']
	>>> s = model.generateSet(10,5) # returns a Set containing 10 traces of length 5
	>>> s.sequences
	[['x', 'a', 'x', 'y', 'a'], ['x', 'b', 'y', 'x', 'a'],
	 ['y', 'b', 'y', 'a', 'x'], ['y', 'b', 'x', 'y', 'b'],
	 ['x', 'b', 'x', 'y', 'a'], ['y', 'b', 'y', 'y', 'x'],
	 ['y', 'b', 'y', 'y', 'y'], ['y', 'a', 'y', 'a', 'y'],
	 ['y', 'a', 'x', 'a', 'y']]
	>>> s.times 
	[1, 1, 1, 1, 1, 1, 2, 1, 1]
	>>> # all the traces appear once in the set, except the 7th which appears twice

Analysis
^^^^^^^^

.. code-block:: python

	>>> model.logLikelihood(s) # loglikelihood of this set of traces under this model
	-4.442498878506513

Saving/Loading
^^^^^^^^^^^^^^

.. code-block:: python

	>>> model.save("my_hmm.txt")
	>>> model2 = ja.loadHMM("my_hmm.txt")

Model
-----

.. autoclass:: jajapy.HMM
   :members:
   :inherited-members:


Other Functions
---------------

.. autofunction:: jajapy.createHMM

.. autofunction:: jajapy.loadHMM

.. autofunction:: jajapy.HMM_random