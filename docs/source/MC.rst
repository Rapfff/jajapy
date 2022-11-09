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
We can create the model depicted above as follow:

.. code-block:: python

	>>> import jajapy as ja
	>>> labeling=['a','b','c','d','a']
	>>> transitions = [(0,1,0.8),(0,2,0.2),
	>>> 		   (1,3,0.6),(1,2,0.4),
	>>> 		   (2,0,0.5),(2,4,0.5),
	>>> 		   (3,2,0.3),(3,3,0.7),
	>>> 		   (4,2,0.2),(4,3,0.1),(4,4,0.7)]
	>>> mc = createMC(transitions, labeling, initial_state=0, name='My_MC')
	>>> print(mc)
	Name: My_MC
	Initial state: s0
	----STATE 0--a----
	s0 -> s1 : 0.8
	s0 -> s2 : 0.2

	----STATE 1--b----
	s1 -> s2 : 0.4
	s1 -> s3 : 0.6

	----STATE 2--c----
	s2 -> s0 : 0.5
	s2 -> s4 : 0.5

	----STATE 3--d----
	s3 -> s2 : 0.3
	s3 -> s3 : 0.7

	----STATE 4--a----
	s4 -> s2 : 0.2
	s4 -> s3 : 0.1
	s4 -> s4 : 0.7
	
We can also generate a random MC

.. code-block:: python

	>>> random_model = ja.MC_random(nb_states=4,
					random_initial_state=False,
					alphabet=['a','b','c'])
	>>> print(random_model)
	Name: MC_random_4_states
	Initial state: s0
	----STATE 0--a----
	s0 -> s0 : 0.2
	s0 -> s1 : 0.26666666666666666
	s0 -> s2 : 0.23333333333333334
	s0 -> s3 : 0.3

	----STATE 1--b----
	s1 -> s0 : 0.28
	s1 -> s1 : 0.36
	s1 -> s2 : 0.04
	s1 -> s3 : 0.32

	----STATE 2--c----
	s2 -> s0 : 0.1875
	s2 -> s1 : 0.125
	s2 -> s2 : 0.1875
	s2 -> s3 : 0.5

	----STATE 3--c----
	s3 -> s0 : 0.5625
	s3 -> s1 : 0.0625
	s3 -> s2 : 0.1875
	s3 -> s3 : 0.1875


Exploration
^^^^^^^^^^^

.. code-block:: python
	
	>>> model.getLabel(0) # label of state 0
	'a'
	>>> model.getLabel(1) # label of state 1
	'b'
	>>> model.tau(0,1,'a') # probability of moving from s0 to s1 seeing 'a' 
	0.8
	>>> model.tau(0,1,'b') # 0.0 since state 0 is not labelled with 'b'
	0.0
	>>> model.a(0,1) # same as model.tau(0,1,'a') since state 0 is labelled with 'a'
	0.8
	>>> model.getAlphabet()	 # all possible observations
	['a','b','c','d']

Running
^^^^^^^

.. code-block:: python

	>>> model.run(5) # returns a list of 5 observations
	['a', 'b', 'd', 'd', 'c']
	>>> s = model.generateSet(10,5) # returns a Set containing 10 traces of size 5
	>>> s.sequences
	[['a', 'b', 'd', 'd', 'd'], ['a', 'b', 'c', 'a', 'b'],
	['a', 'b', 'd', 'c', 'a'], ['a', 'b', 'd', 'd', 'c']]
	>>> s.times # the first sequence appears four times, the second twice, etc...
	[4, 2, 3, 1]

Analysis
^^^^^^^^

.. code-block:: python

	>>> model.logLikelihood(s) # loglikelihood of this set of traces under this model
	-1.8009169143518982

Saving/Loading
^^^^^^^^^^^^^^

.. code-block:: python

	>>> model.save("my_mc.txt")
	>>> same_model = ja.loadMC("my_mc.txt")

Model
-----

.. autoclass:: jajapy.MC
   :members:
   :inherited-members:

Other Functions
---------------
.. autofunction:: jajapy.createMC

.. autofunction:: jajapy.loadMC

.. autofunction:: jajapy.MC_random