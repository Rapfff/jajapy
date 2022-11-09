Tutorial
===============

1. A simple example with HMMs
-----------------------------

:download:`python file <https://github.com/Rapfff/jajapy/tree/main/examples/01-hmms.py>`

In this example, we will:

1. Create a HMM *H* from scratch,
2. Use it to generate a training set,
3. Use the Baum-Welch algorithm to learn, from the training set, *H*,
4. Compare *H* with the model generated at the previous step.

Creating a HMM
^^^^^^^^^^^^^^

.. image:: pictures/HMM.png
	:width: 60%
	:align: center

.. _create-hmm-example:

We can create the model depicted above like this:

.. code-block:: python

	import jajapy as ja
	from numpy import array
	alphabet = ['a','b','x','y']
	nb_states = 5

	# in the next state we generate 'x' with probability 0.4, and 'y' with probability 0.6
	# once an observation generated, we move to state 1 or 2 with probability 0.5.
	s0 = ja.HMM_state([("x",0.4),("y",0.6)],[(1,0.5),(2,0.5)],alphabet,nb_states)
	s1 = ja.HMM_state([("a",0.8),("b",0.2)],[(3,1.0)],alphabet,nb_states)
	s2 = ja.HMM_state([("a",0.1),("b",0.9)],[(4,1.0)],alphabet,nb_states)
	s3 = ja.HMM_state([("x",0.5),("y",0.5)],[(0,0.8),(1,0.1),(2,0.1)],alphabet,nb_states)
	s4 = ja.HMM_state([("y",1.0)],[(3,1.0)],alphabet,nb_states)
	transitions = array([s0[0],s1[0],s2[0],s3[0],s4[0]])
	output = array([s0[1],s1[1],s2[1],s3[1],s4[1]])
	original_model = ja.HMM(transitions,output,alphabet,initial_state=0,name="My HMM")
	print(original_model)

*(optional)* This model can be saved into a text file and then loaded as follow:

.. code-block:: python

	original_model.save("my_model.txt")
	original_model = ja.loadHMM("my_model.txt")


Generating a training set
^^^^^^^^^^^^^^^^^^^^^^^^^
Now we can generate a training set. This training set contains 1000 traces, which all consists of 10 observations.

.. code-block:: python

	# We generate 1000 sequences of 10 observations
	training_set = original_model.generateSet(set_size=1000, param=10)

*(optional)* This Set can be saved into a text file and then loaded as follow:

.. code-block:: python

	training_set.save("my_training_set.txt")
	training_set = ja.loadSet("my_training_set.txt")


Learning a HMM using BW
^^^^^^^^^^^^^^^^^^^^^^^
Let now use our training set to learn ``original_model`` with the Baum-Welch algorithm:

.. code-block:: python

	output_model = ja.BW_HMM().fit(training_set, nb_states=5, stormpy_output=False)
	print(output_model)

For the initial model we used a randomly generated HMM with 5 states. Since we are not planning to use Storm on this model,
we set the `stormpy_output` parameter to False.

Evaluating the BW output model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Eventually we compare the output model with the original one. The usual way to do so is to generate a test set and compare
the loglikelihood of it under each of the two models. As the training set, our test set will contain 1000 traces of length 10.

.. code-block:: python

	# We generate 1000 sequences of 10 observations
	test_set = original_model.generateSet(set_size=1000, param=10)

Now we can compute the loglikelihood under each model:

.. code-block:: python

	ll_original = original_model.logLikelihood(test_set)
	ll_output   =   output_model.logLikelihood(test_set)
	quality = ll_original - ll_output
	print(quality)

If ``quality`` is positive then we are overfitting.


2. An example with MC: random restart
-------------------------------------

:download:`python file <https://github.com/Rapfff/jajapy/tree/main/examples/02-mcs.py>`


This time we will try to learn the `Reber grammar <https://cnl.salk.edu/~schraudo/teach/NNcourse/reber.html>`_.
We have added probabilities on the transitions in order to have a MC.

.. image:: pictures/REBER.png
	:width: 80%
	:align: center

As before we will first create the original model and generate the training set, then we will learn it several times
with different random initial hypothesis. We will keep only the best model, i.e. the one maximizing the loglikelihood
of the test set. This technique is called *random restart*.

Creating the MC and generating the training set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This step is similar to what we did before.

.. code-block:: python

	>>> import jajapy as ja
	>>> def modelMC_REBER():
	...		# State 0 is labelled with B, state 1 with T, etc...
	...		labeling = list("BTSXSPTXPVVE")
	...		initial_state = 0
	...		name = "MC_REBER"
	...		# From state 0 we move to state 1 with probability 0.5
	...		# and to state 5 with probability 0.5
	...		transitions = [(0,1,0.5),(0,5,0.5),(1,2,0.6),(1,3,0.4),(2,2,0.6),(2,3,0.4),
	...			       (3,7,0.5),(3,4,0.5),(4,11,1.0),(5,6,0.7),(5,9,0.3),
	...			       (6,6,0.7),(6,9,0.3),(7,6,0.7),(7,9,0.3),(8,7,0.5),(8,4,0.5),
	...			       (9,8,0.5),(9,10,0.5),(10,11,1.0),(11,11,1.0)]
	...		return ja.createMC(transitions,labeling,initial_state,name)
	>>> original_model = modelMC_REBER()
	>>> training_set = original_model.generateSet(1000,10)
	>>> test_set = original_model.generateSet(1000,10)

Learning a MC using random restart
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We will learn the model 10 times and keep only the best one (according to the test set loglikelihood).

.. code-block:: python

	>>> nb_trials = 10

At each iteration, the library will generate a new model with 7 states.

.. code-block:: python

	>>> best_model = None
	>>> quality_best = -1024
	>>> for n in range(1,nb_trials+1):
	...		current_model = ja.BW_MC().fit(training_set,nb_states=7,pp=n, stormpy_output=False)
	...		current_quality = current_model.logLikelihood(test_set)
	...		if quality_best < current_quality: #we keep the best model only
	...			quality_best = current_quality
	...			best_model = current_model

	1 |████████████████████████████████████████| (!) 15 in 1.1s (13.32/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   15
	Running time:	   1.148518
	---------------------------------------------

	2 |████████████████████████████████████████| (!) 17 in 1.3s (13.27/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   17
	Running time:	   1.282217
	---------------------------------------------

	3 |████████████████████████████████████████| (!) 15 in 1.1s (13.29/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   15
	Running time:	   1.130267
	---------------------------------------------

	4 |████████████████████████████████████████| (!) 15 in 1.1s (13.11/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   15
	Running time:	   1.145003
	---------------------------------------------

	5 |████████████████████████████████████████| (!) 20 in 1.4s (13.85/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   20
	Running time:	   1.445236
	---------------------------------------------

	6 |████████████████████████████████████████| (!) 13 in 1.0s (13.48/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   13
	Running time:	   0.966637
	---------------------------------------------

	7 |████████████████████████████████████████| (!) 19 in 1.4s (13.25/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   19
	Running time:	   1.435106
	---------------------------------------------

	8 |████████████████████████████████████████| (!) 11 in 0.8s (13.31/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   11
	Running time:	   0.828422
	---------------------------------------------

	9 |████████████████████████████████████████| (!) 16 in 1.2s (13.11/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   16
	Running time:	   1.222533
	---------------------------------------------

	10 |████████████████████████████████████████| (!) 14 in 1.1s (13.19/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   14
	Running time:	   1.062477
	---------------------------------------------


Notice that the current trial number appears at the beginnig of each print: this is because we 
have set the ``pp`` parameter of the ``fit`` method with the current trial number.

.. code-block:: python

	>>> print(quality_best)
	-4.649108177287

The loglikelihood of the test set under the best model is good. Let's have a look to the model:

.. code-block:: python

	>>> print(best_model)
	Name: unknown_MC
	Initial state: s0
	----STATE 0--B----
	s0 -> s1 : 0.507
	s0 -> s2 : 0.493

	----STATE 1--P----
	s1 -> s6 : 0.17521902311933552
	s1 -> s8 : 0.1902377448236495
	s1 -> s10 : 0.18022528160197424
	s1 -> s11 : 0.45431789737171446

	----STATE 2--T----
	s2 -> s5 : 0.43872597184243073
	s2 -> s7 : 0.0014362999628431126
	s2 -> s9 : 0.5598377281947261

	----STATE 3--V----
	s3 -> s4 : 0.9999999999308414

	----STATE 4--E----
	s4 -> s4 : 0.9999999999711968

	----STATE 5--X----
	s5 -> s6 : 0.5517000753834211
	s5 -> s8 : 0.44829992460404955

	----STATE 6--S----
	s6 -> s4 : 0.9999999999985493

	----STATE 7--X----
	s7 -> s6 : 0.6098517550064492
	s7 -> s8 : 0.3901482394741814

	----STATE 8--X----
	s8 -> s10 : 0.2910662820008986
	s8 -> s11 : 0.708933717432188

	----STATE 9--S----
	s9 -> s5 : 0.40493202909358567
	s9 -> s7 : 0.0010381201601459317
	s9 -> s9 : 0.594029850746269

	----STATE 10--V----
	s10 -> s1 : 0.484490398807296
	s10 -> s3 : 0.515509601192651

	----STATE 11--T----
	s11 -> s10 : 0.3041301466409757
	s11 -> s11 : 0.6958698371678799


One can be suprised to see that the probability to leave *s4* is not equal to one.
The reason is that *jajapy* doesn't print out the transitions with a very low probability,
for a better readability.  

3. An example with MDP: active learning and model checking
----------------------------------------------------------
:download:`python file <https://github.com/Rapfff/jajapy/tree/main/examples/03-mdps.py>`

Here, we will learn a MDP representing the following grid world:


.. image:: pictures/grid.png
	:width: 40%
	:align: center

We start in the top-left cell and our destination is the bottom-right one.
We can move in any of the four directions *North, South, East and West*.
We may make errors in movement, e.g. move south west instead of south with
an error probability depending on the target terrain. This model is the one
in `this paper <https://arxiv.org/pdf/2110.03014.pdf>`_.

First we create the original model.

.. code-block:: python

	import jajapy as ja
	from numpy import array

	def modelMDP_gridworld():
		alphabet = ['S','M','G','C','W',"done"]
		actions = list("nsew")
		nb_states = 9
		s0 = ja.MDP_state({'n': [(0,'W',1.0)],
						's': [(3,'M',0.6),(4,'G',0.4)],
						'e': [(1,'M',0.6),(4,'G',0.4)],
						'w': [(0,'W',1.0)]
						},alphabet,nb_states,actions)
		s1 = ja.MDP_state({'n': [(1,'W',1.0)],
						's': [(4,'G',0.8),(3,'M',0.1),(5,'C',0.1)],
						'e': [(2,'G',0.8),(5,'C',0.2)],
						'w': [(0,'S',0.75),(3,'M',0.25)]
						},alphabet,nb_states,actions)
		s2 = ja.MDP_state({'n': [(2,'W',1.0)],
						's': [(5,'C',1.0)],
						'e': [(2,'W',1.0)],
						'w': [(1,'M',0.6),(4,'G',0.4)]
						},alphabet,nb_states,actions)
		s3 = ja.MDP_state({'n': [(0,'S',0.75),(1,'M',0.25)],
						's': [(6,'G',0.8),(7,'S',0.2)],
						'e': [(4,'G',0.8),(1,'M',0.1),(7,'S',0.1)],
						'w': [(3,'M',1.0)]
						},alphabet,nb_states,actions)
		s4 = ja.MDP_state({'n': [(1,'M',0.6),(0,'S',0.2),(2,'G',0.2)],
						's': [(7,'S',0.75),(6,'G',0.125),(8,'done',0.125)],
						'e': [(5,'C',1.0)],
						'w': [(3,'M',0.6),(0,'S',0.2),(6,'G',0.2)]
						},alphabet,nb_states,actions)
		s5 = ja.MDP_state({'n': [(2,'G',0.8),(1,'M',0.2)],
						's': [(8,'done',0.6),(7,'S',0.4)],
						'e': [(5,'W',1.0)],
						'w': [(4,'G',0.8),(1,'M',0.1),(7,'S',0.1)]
						},alphabet,nb_states,actions)
		s6 = ja.MDP_state({'n': [(3,'M',0.6),(4,'G',0.4)],
						's': [(6,'W',1.0)],
						'e': [(7,'S',0.75),(4,'G',0.25)],
						'w': [(6,'W',1.0)]
						},alphabet,nb_states,actions)
		s7 = ja.MDP_state({'n': [(1,'M',0.6),(0,'S',0.2),(2,'G',0.2)],
						's': [(7,'W',1.0)],
						'e': [(8,'done',0.6),(5,'C',0.4)],
						'w': [(6,'G',0.8),(3,'M',0.2)]
						},alphabet,nb_states,actions)
		s8 = ja.MDP_state({'n': [(8,'done',1.0)],
						's': [(8,'done',1.0)],
						'e': [(8,'done',1.0)],
						'w': [(8,'done',1.0)]
						},alphabet,nb_states,actions)
		matrix = array([s0,s1,s2,s3,s4,s5,s6,s7,s8])
		return ja.MDP(matrix,alphabet,actions,initial_state=0,name="grid world")

Then we generate our training set and test set. Since MDPs are non-deterministic, we need to specify to
jajapy which scheduler we want it to use to generate training/test sets. Here we will
use uniform scheduler (all the actions have the same probability to be chosen).

.. code-block:: python

	original_model = modelMDP_gridworld()
	# SETS GENERATION
	#------------------------
	# We generate 1000 sequences of 10 observations for each set
	scheduler = ja.UniformScheduler(original_model.getActions())
	training_set = original_model.generateSet(1000,10,scheduler)
	test_set = original_model.generateSet(1000,10,scheduler)

Then we can learn the model. Here we do 20 active learning iterations:
for each of them we generate 50 new sequences. These sequences will be generated
using the *active learning scheduler* with probability 0.75, and with a uniform
scheduler with probability 0.25. A learning rate equal to zero means here that 
we will learn, at each iteration, using all the sequences we have (i.e. using the sequences
from the original training set plus the one generated by all the previous active learning
iterations).

.. code-block:: python

	# LEARNING
	#---------
	learning_rate = 0
	output_model = ja.Active_BW_MDP().fit(training_set,learning_rate,
								nb_iterations=20, nb_sequences=50,
								epsilon_greedy=0.75, nb_states=9,
								stormpy_output=False)
	output_quality = output_model.logLikelihood(test_set)
	
	print(output_model)
	print(output_quality)

.. _stormpy-example:

Now, one could ask for the scheduler which minimizes the number of step before reaching our objective,
the bottom-right state. For this, we can use stormpy:

.. code-block:: python

	# MODEL CHECKING
	#---------------
	storm_model = ja.jajapyModeltoStorm(output_model)
	properties = stormpy.parse_properties("Rmax=? [ F \"done\" ]")
	result = stormpy.check_model_sparse(storm_model, properties[0], extract_scheduler=True)
	scheduler = result.scheduler


4. An advanced example with MC and model checking
-------------------------------------------------

:download:`python file <https://github.com/Rapfff/jajapy/tree/main/examples/04-mcs_with_stormpy.py>`


.. image:: pictures/knuthdie.png
	:width: 60%
	:align: center

In this example, we will first learn a MC representation of the Yao-Knuth'die 
(see above) using some structural knowledge we have. Then, we will use *stormpy* to check
if our model satisfies some properties.

As usual, we start by creating the training and test set.

.. code-block:: python

	import jajapy as ja
	from numpy import array

	def modelMC_KnuthDie(p=0.5):
		alphabet = ["P","F","one","two","three","four","five","six"]
		nb_states = 13
		s0 = ja.MC_state([(1 ,'P',p),(2 ,'F',1-p)],alphabet,nb_states)
		s1 = ja.MC_state([(3 ,'P',p),(4 ,'F',1-p)],alphabet,nb_states)
		s2 = ja.MC_state([(5 ,'P',p),(6 ,'F',1-p)],alphabet,nb_states)
		s3 = ja.MC_state([(1 ,'P',p),(7 ,'F',1-p)],alphabet,nb_states)
		s4 = ja.MC_state([(8 ,'P',p),(9 ,'F',1-p)],alphabet,nb_states)
		s5 = ja.MC_state([(10,'P',p),(11,'F',1-p)],alphabet,nb_states)
		s6 = ja.MC_state([(12,'P',p),(2 ,'F',1-p)],alphabet,nb_states)
		s7 = ja.MC_state([(7 ,  'one',1.0)],alphabet,nb_states)
		s8 = ja.MC_state([(8 ,  'two',1.0)],alphabet,nb_states)
		s9 = ja.MC_state([(9 ,'three',1.0)],alphabet,nb_states)
		s10= ja.MC_state([(10, 'four',1.0)],alphabet,nb_states)
		s11= ja.MC_state([(11, 'five',1.0)],alphabet,nb_states)
		s12= ja.MC_state([(12,  'six',1.0)],alphabet,nb_states)
		matrix = array([s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12])
		return ja.MC(matrix,alphabet,initial_state=0,name="Knuth's Die")
	
	original_model = modelMC_KnuthDie()
	# SETS GENERATION
	#------------------------
	# We generate 1000 sequences of 10 observations for each set
	training_set = original_model.generateSet(1000,10)
	test_set = original_model.generateSet(1000,10)

Now, we can learn the model using Baum-Welch. But here, we assume that we have some knowledge about
the structure of what we are learning. In fact, Baum-Welch improve the initial model iteratively by
removing some transitions and changing some transitions probabilities, but it cannot create a new
transition: if there is no transition between *s0* and *s1* in the initial hypothesis, there will be
no transition there as well in the output model. Let say that here we know that what we are learning
looks like this (we don't have any information about the transitions in the shaded area):

.. image:: pictures/knuthdie_hint.png
	:width: 60%
	:align: center



We can now create our initial hypothesis and learn the model. Once again, we will use random restart
to keep only the best model we get.

.. code-block:: python

	def firstGuess():
		alphabet = ["P","F","one","two","three","four","five","six"]
		nb_states = 13
		s0 = ja.MC_state(list(zip([1,2],['P','F'],ja.randomProbabilities(2))),alphabet,nb_states)
		s1 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6],
								['P','F','P','F','P','F','P','F','P','F','P','F'],
								ja.randomProbabilities(12))),
						alphabet,nb_states)
		s2 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6],
								['P','F','P','F','P','F','P','F','P','F','P','F'],
								ja.randomProbabilities(12))),
						alphabet,nb_states)
		s3 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12],
								['P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',
								'P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',],
								ja.randomProbabilities(24))),
						alphabet,nb_states)

		s4 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12],
								['P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',
								'P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',],
								ja.randomProbabilities(24))),
						alphabet,nb_states)

		s5 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12],
								['P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',
								'P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',],
								ja.randomProbabilities(24))),
						alphabet,nb_states)

		s6 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12],
								['P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',
								'P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',],
								ja.randomProbabilities(24))),
						alphabet,nb_states)
		s7 = ja.MC_state([(7 ,  'one',1.0)],alphabet,nb_states)
		s8 = ja.MC_state([(8 ,  'two',1.0)],alphabet,nb_states)
		s9 = ja.MC_state([(9 ,'three',1.0)],alphabet,nb_states)
		s10= ja.MC_state([(10, 'four',1.0)],alphabet,nb_states)
		s11= ja.MC_state([(11, 'five',1.0)],alphabet,nb_states)
		s12= ja.MC_state([(12,  'six',1.0)],alphabet,nb_states)
		matrix = array([s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12])
		return ja.MC(matrix,alphabet,initial_state=0,name="first guess")
	
	# LEARNING
	#---------
	nb_trials = 10 # we will repeat learn this model 10 times
	best_model = None
	quality_best = -1024
	for n in range(1,nb_trials+1):
		current_model = ja.BW_MC().fit(training_set,initial_model=firstGuess())
		current_quality = current_model.logLikelihood(test_set)
		if quality_best < current_quality: #we keep the best model only
				quality_best = current_quality
				best_model = current_model

	print(quality_best)
	print(best_model)

Now, we would like to check if we have a probability of 1/6 to get a *"five"* with 
this new model.

.. code-block:: python

	# MODEL CHECKING
	#---------------
	model_storm = ja.jajapyModeltoStorm(best_model)
	formula_str = 'P=? [F "five"]'
	properties = stormpy.parse_properties(formula_str)
	result = stormpy.check_model_sparse(model_storm,properties[0])
	print(result.at(model_storm.initial_states[0]))

