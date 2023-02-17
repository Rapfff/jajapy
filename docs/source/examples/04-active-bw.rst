.. _example-active-bw :

4. Learning an MDP using Active-BW
==================================

:download:`python file <https://github.com/Rapfff/jajapy/tree/main/examples/04-active-bw.py>`
:download:`prism model for the grid <https://github.com/Rapfff/jajapy/tree/main/examples/materials/grid_3x3.sm>`

For this example, we will learn the grid world model depicted below using the active learning extension
introduced `this paper <https://arxiv.org/pdf/2110.03014.pdf>`_.


.. image:: ../pictures/grid_3x3.png
	:width: 40%
	:align: center

We start in the top-left cell and our destination is the bottom-right one.
We can move in any of the four directions *North, South, East and West*.
We may make errors in movement, e.g. move south west instead of south with
an error probability depending on the target terrain. 

This model is form `this paper <https://arxiv.org/pdf/2110.03014.pdf>`_.

Loading the original model from a Prism file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The original model is described in a Prism file. Let's first load it.

.. code-block:: python

	>>> import jajapy as ja
	>>> actions = list('nsew')
	>>> labels = list("SMGCW")+["GOAL"]
	>>> original_model = ja.loadPrism('materials/grid_3x3.sm')
	>>> original_model.actions = actions # otherwise the actions are a0, a1, etc...

The last line matters: in fact, the actions name are lost in the process, and replaced
by *a0, a1, a2, a3*. This problem will be (hopefully) solved in the next releases.

You may notice, if you try this on your machine, that the command ``loadPrism`` clears the terminal.
The reason is that, loading a Prism file, causes many warnings/errors prints, even if, at the end,
everything went well (yes,I know it is hard to believe, but trust me). Hence, to avoid panic,
the command clear the terminal and the user is serene.

Generating the training set
^^^^^^^^^^^^^^^^^^^^^^^^^^^
We can now generate the training. However, since MDPs are non-deterministic models, we need a
scheduler to resolve the non-deterministic choices. We will use here an uniform scheduler.

Here, we generate a training set with 1,000 traces of length 10.

.. code-block:: python

	>>> # We generate 1000 sequences of 10 observations
	>>> # using an uniform scheduler to resolve the non-deterministic choices.
	>>> training_set1= original_model.generateSet(1000,10,scheduler=ja.UniformScheduler(actions))

Generating the initial hypothesis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The system under learning contains 16 states, and only 6 labels. Hence, if we let *Jajapy* generate
a random MDP with 16 states for the training set, the first 6 states will be labeled with *S, C, M, G, W*
and *GOAL*, and the 10 remaining will be labeled randomly. Hence, we could possibly have 11 states labeled
with *GOAL* and only one with *W*, which is far away from what we have in the system under learning.

Here, we will first randomly generate our initial hypothesis, and then modify its labeling to have an initial
hypothesis closer to the system under learning.

.. code-block:: python

	>>> initial_hypothesis = ja.MDP_random(nb_states=16,labeling=labels,actions=actions,random_initial_state=False)
	WARNING: the size of the labeling is lower than the number of states. The labels for the last states will be chosen randomly.
	>>> initial_hypothesis.labeling = original_model.labeling


.. note::

	Before doing that, we must be sure that the *init* label is at the same index in both ``initial_hypothesis.labeling`` and
	``original_model.labeling``, and that they both have the same length. Here, we now that our initial hypothesis has as many
	state as the original model, thus the two list have the same length. And we know that the *init* label is the last one in
	these two lists.


Active Learning
^^^^^^^^^^^^^^^
The active learning starts by one BW execution on the given training set
and initial hypothesis, and then executes several BW executions (``nb_iterations=10``) on the current hypothesis
and a training set containing ``training_set`` plus a number of sequences (here ``nb_sequences=100``) generated
by the original model with the *active learning scheduler* described in `the paper <https://arxiv.org/pdf/2110.03014.pdf>`_.
In practice, for each new sequence, it uses the *active learning scheduler* with probability 0.9 and an uniform scheduler
with probability 0.1. These probabilities can be changed by setting the parameter ``epsilon_greedy`` of the ``fit``
method. The legnth of the additional traces can be set using the ``sequence_length`` parameter.

.. code-block:: python

	>>> output_model_active  = ja.Active_BW_MDP().fit(training_set1, lr=0, nb_iterations=10,
							nb_sequences=100,initial_model=initial_hypothesis)
	Passive iteration: |████████████████████████████████████████| (!) 28 in 41.4s (0.68/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   28
	Running time:	   41.411273
	---------------------------------------------

	Active iteration 1/10:  |████████████████████████████████████████| (!) 3 in 4.9s (0.61/s) 
	Active iteration 2/10:  |████████████████████████████████████████| (!) 3 in 4.9s (0.61/s) 
	Active iteration 3/10:  |████████████████████████████████████████| (!) 3 in 5.2s (0.58/s) 
	Active iteration 4/10:  |████████████████████████████████████████| (!) 5 in 9.1s (0.55/s) 
	Active iteration 5/10:  |████████████████████████████████████████| (!) 10 in 19.4s (0.52/s) 
	Active iteration 6/10:  |████████████████████████████████████████| (!) 25 in 49.6s (0.50/s) 
	Active iteration 7/10:  |████████████████████████████████████████| (!) 3 in 6.2s (0.48/s) 
	Active iteration 8/10:  |████████████████████████████████████████| (!) 3 in 6.5s (0.46/s) 
	Active iteration 9/10:  |████████████████████████████████████████| (!) 3 in 6.9s (0.44/s) 
	Active iteration 10/10:  |████████████████████████████████████████| (!) 23 in 53.8s (0.43/s) 

	---------------------------------------------
	Learning finished
	Iterations:	   109
	Running time:	   213.37684
	---------------------------------------------

Model checking and evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We can now model check the output model and compare the results with the original one.

.. code-block:: python

	>>> import stormpy
	>>> formulas = ["Pmax=? [ F<=3 \"GOAL\"  ]","Pmax=? [ !(\"C\"|\"W\") U<=8\"GOAL\" ]", "Pmax=? [ F<=5 \"GOAL\"  ]"]
	>>> original_model = ja.jajapyModeltoStormpy(original_model)
	>>> for formula in formulas:
	>>> 	properties = stormpy.parse_properties(formula)
	>>> 	result_original = stormpy.check_model_sparse(original_model, properties[0])
	>>> 	result_original = result_original.at(original_model.initial_states[0])
	>>> 	result_output_active = stormpy.check_model_sparse(output_model_active, properties[0])
	>>> 	result_output_active = result_output_active.at(output_model_active.initial_states[0])
	>>> 	print(formula,'in the original model:',str(result_original))
	>>> 	print(formula,'in the output model active:',str(result_output_active))
	>>> 	print()
	Pmax=? [ F<=3 "GOAL"  ] in the original model: 0.049999999999999996
	Pmax=? [ F<=3 "GOAL"  ] in the output model active: 0.05134150997867366

	Pmax=? [ !("C"|"W") U<=6"GOAL" ] in the original model: 0.6291224999999998
	Pmax=? [ !("C"|"W") U<=6"GOAL" ] in the output model active: 0.6653982274619132

	Pmax=? [ F<=5 "GOAL"  ] in the original model: 0.7247999999999999
	Pmax=? [ F<=5 "GOAL"  ] in the output model active: 0.7172809998333902

