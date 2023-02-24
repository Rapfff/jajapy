.. _example-ioalergia :

10. Learning an MDP with IOAlergia
==================================

:download:`python file <https://github.com/Rapfff/jajapy/tree/main/examples/10-ioalergia.py>`


This time we will use `IOAlergia <https://people.cs.aau.dk/~tdn/papers/ML_Hua.pdf>` to learn an MDP.

.. image:: ../pictures/grid_3x3.png
	:width: 40%
	:align: center

We start in the top-left cell and our destination is the bottom-right one.
We can move in any of the four directions *North, South, East and West*.
We may make errors in movement, e.g. move south west instead of south with
an error probability depending on the target terrain. 

This model is form `this paper <https://arxiv.org/pdf/2110.03014.pdf>`_.


Creating the original MDP and generating the training set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This step is similar to the first steps in :ref:`example-active-bw`.

.. code-block:: python

	>>> import jajapy as ja
	>>> actions = list('nsew')
	>>> labels = list("SMGCW")+["GOAL"]
	>>> original_model = ja.loadPrism('materials/grid_3x3.sm')
	>>> original_model.actions = actions # otherwise the actions are a0, a1, etc...
	>>> # We generate 1000 sequences of 10 observations
	>>> # using an uniform scheduler to resolve the non-deterministic choices.
	>>> training_set1= original_model.generateSet(1000,10,scheduler=ja.UniformScheduler(actions))

Learning the MC using Alergia
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We can now learn the model.

.. code-block:: python

	>>> ioalergia_model = ja.IOAlergia().fit(training_set,epsilon=0.5,stormpy_output=False)

We set ``stormpy_output`` to ``False`` since we want to compute some traces loglikelihood
afterward.

Output model evaluation
^^^^^^^^^^^^^^^^^^^^^^^

First, we notice that the output model is much bigger than the original one.
With a smaller ``epsilon`` parameter we would have gotten a smaller output model.

.. code-block:: python
	
	>>> print(original.nb_states)
	17
	>>> print(ioalergia_model.nb_states)
	60

Let's evaluate our model under the test set:

.. code-block:: python

	>>> print('Loglikelihood for the original model    :',original_model.logLikelihood(test_set))
	Loglikelihood for the original model    : -3.5086078777607406
	>>> print('Loglikelihood for IOAlergia output model:',ioalergia_model.logLikelihood(test_set))
	Loglikelihood for IOAlergia output model: -8.586780409259692


And now under some properties. But first, we need to translate our *Jajapy* models to *Stormpy* sparse models.

.. code-block:: python
	
	>>> ioalergia_model = ioalergia_model.toStormpy()
	>>> original_model = original_model.toStormpy()
	>>> #
	>>> formulas = ["Pmax=? [ F<=3 \"GOAL\"  ]","Pmax=? [ !(\"C\"|\"W\") U<=6\"GOAL\" ]", "Pmax=? [ F<=5 \"GOAL\"  ]"]
	>>> for formula in formulas:
	>>> 	properties = stormpy.parse_properties(formula)
	>>> 	result_original = stormpy.check_model_sparse(original_model, properties[0])
	>>> 	result_original = result_original.at(original_model.initial_states[0])
	>>> 	result_ioalergia = stormpy.check_model_sparse(ioalergia_model, properties[0])
	>>> 	result_ioalergia = result_ioalergia.at(ioalergia_model.initial_states[0])
	>>> 	print(formula,'in the original model:',str(result_original))
	>>> 	print(formula,'in the output model active:',str(result_ioalergia))
	>>> 	print()
	Pmax=? [ F<=3 "GOAL"  ] in the original model: 0.049999999999999996
	Pmax=? [ F<=3 "GOAL"  ] in the output model active: 0.11055194805194804

	Pmax=? [ !("C"|"W") U<=6"GOAL" ] in the original model: 0.6291224999999998
	Pmax=? [ !("C"|"W") U<=6"GOAL" ] in the output model active: 0.58839483993125

	Pmax=? [ F<=5 "GOAL"  ] in the original model: 0.7247999999999999
	Pmax=? [ F<=5 "GOAL"  ] in the output model active: 0.7739474791032257
