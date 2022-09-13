Markov Decision Process (MDP)
=============================
A MDP is similar to a MC, but with non-determinism. More information `here <https://en.wikipedia.org/wiki/Markov_decision_process>`_.

Example
-------

In the following picture the actions are represented by the colors. Hence, by executing action *blue* in the leftern state, we will, 
with probability one, observe *a* and reach the bottom state.

.. image:: pictures/MDP.png
   :width: 75 %
   :align: center

Creation
^^^^^^^^
.. code-block:: python

	>>> import jajapy as ja
	>>> from numpy import array
	>>> alphabet = ['a','b','c','d']
	>>> actions  = ['red', 'blue']
	>>> nb_states = 3
	>>> s0 = ja.MDP_state({'red': [(1,'a',0.5),(2,'a',0.5)], 'blue': [(2,'a',1.0)]}, alphabet, nb_states, actions)
	>>> s1 = ja.MDP_state({'red': [(2,'b',1.0),(2,'a',0.5)], 'blue': [(1,'c',1.0)]}, alphabet, nb_states, actions)
	>>> s2 = ja.MDP_state({'blue': [(2,'d',1.0)]}, alphabet, nb_states, actions)
	>>> lst_states = array([s0, s1, s2])
	>>> model = ja.MDP(lst_states,alphabet,actions,initial_state=0,name="My MDP")


We can also generate a random MDP

.. code-block:: python

	>>> random_model = ja.MDP_random(number_states=5,
					random_initial_state=False,
					alphabet=['a','b','c','d'],
					actions=['red','blue'])

Exploration
^^^^^^^^^^^

.. code-block:: python

	>>> model.tau(0,'red',1,'a') # probability of moving from s0 to s1 seeing 'a' after executing 'red'
	0.5
	>>> model.observations() # all possible observations
	['a','b','c','d']
	>>> model.states[0].observations() # all possible observations in s0
	['a']
	>>> model.actions() # all possible actions
	['red','blue']
	>>> model.actions(2) # all actions available in state s2
	['blue']

Running
^^^^^^^

.. code-block:: python

	>>> scheduler = ja.UniformScheduler(model.actions())
	>>> model.run(5,scheduler) # returns a list of 5 actions and 5 observations
	['blue', 'a', 'blue', 'd', 'blue', 'd', 'blue', 'd', 'blue', 'd']
	>>> s = model.generateSet(10,5,scheduler) # returns a Set containing 10 traces of size 5
	>>> s.sequences
	[['blue', 'a', 'blue', 'd', 'blue', 'd', 'blue', 'd', 'blue', 'd'],
	 ['red', 'a', 'blue', 'c', 'red', 'b', 'blue', 'd', 'blue', 'd'],
	 ['red', 'a', 'red', 'b', 'blue', 'd', 'blue', 'd', 'blue', 'd'],
	 ['red', 'a', 'blue', 'd', 'blue', 'd', 'blue', 'd', 'blue', 'd']]
	>>> s.times # first sequence appears 6 times, the second twice, etc...
	[6, 2, 1, 1]

Analysis
^^^^^^^^

.. code-block:: python

	>>> model.logLikelihood(s) # loglikelihood of this set of traces under this model
	-0.2772588722239781

Saving/Loading
^^^^^^^^^^^^^^

.. code-block:: python

	>>> model.save("my_mdp.txt")
	>>> another_model = ja.loadMDP("my_mdp.txt")
	
Model
-----

.. autoclass:: jajapy.MDP
   :members:
   :inherited-members:

Other Functions
---------------
.. autofunction:: jajapy.MDP_state

.. autofunction:: jajapy.loadMDP

.. autofunction:: jajapy.MDP_random

.. autofunction:: jajapy.MDPFileToPrism

.. autofunction:: jajapy.loadPrismMDP

Scheduler
---------
Scheduler are used to solve the non-deterministic choices while running a MDP.
A scheduler will basically chooses the actions to execute at each time step.

There are several kind of scheduler. The easiest one in the *uniform* one, which,
at each time step, chooses an action uniformly at random. The *memoryless* scheduler
will choose the action according to the current state of the MDP (thus it assumes that
the MDP states are observable). It basically maps each MDP state to one action. Finally
a *finite memory scheduler* can be seen as a generative automaton that takes on input
a sequence of observation and returns a sequence of actions.

.. autoclass:: jajapy.UniformScheduler
   :members:
   :inherited-members:

.. autoclass:: jajapy.MemorylessScheduler
   :members:
   :inherited-members:

.. autoclass:: jajapy.FiniteMemoryScheduler
   :members:
   :inherited-members: