Getting Started
===============

A simple example with HMMs
--------------------------

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

We can create the model depicted above like this:

.. code-block:: python

	import jajapy as ja
	# in the next state we generate 'x' with probaility 0.4, and 'y' with probability 0.6
	# once an observation generated, we move to state 1 or 2 with probability 0.5
	# the id of this state is 0.
	s0 = ja.HMM_state([[0.4,0.6],['x','y']], [[0.5,0.5],[1,2]], 0)
	# same logic for the other states.
	s1 = ja.HMM_state([[0.8,0.2],['a','b']], [[1.0],[3]], 1)
	s2 = ja.HMM_state([[0.1,0.9],['a','b']], [[1.0],[4]], 2)
	s3 = ja.HMM_state([[0.5,0.5],['x','y']], [[0.8,0.1,0.1],[0,1,2]], 3)
	s4 = ja.HMM_state([[1.0],['y']], [[1.0],[3]], 4)
	lst_states = [s0, s1, s2, s3, s4]
	original_model = ja.HMM(states=lst_states,initial_state=0,name="My HMM")
	print(original_model)

*(optional)* This model can be saved into a text file and then loaded as follow:

.. code-block:: python

	original_model.save("my_model.txt")
	original_model = ja.loadHMM("my_model.txt")


Generating a training set
^^^^^^^^^^^^^^^^^^^^^^^^^
Now we can generate a training set. This training set contains 1000 traces, which all consists of 10 observations.

.. code-block:: python

	training_set = original_model.generateSet(set_size=1000, param=10)

*(optional)* This Set can be saved into a text file and then loaded as follow:

.. code-block:: python

	training_set.save("my_training_set.txt")
	training_set = ja.loadSet("my_training_set.txt")


Learning a HMM using BW
^^^^^^^^^^^^^^^^^^^^^^^
Let now use our training set to learn ``original_model`` with the Baum-Welch algorithm:

.. code-block:: python

	output_model = ja.BW_HMM().fit(training_set, nb_states=5)
	print(output_model)

For the initial model we used a randomly generated HMM with 5 states.

Evaluating the BW output model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Eventually we compare the output model with the original one. The usual way to do so is to generate a test set and compare
the loglikelihood of it under each of the two models. As the training set, our test set will contain 1000 traces of length 10.

.. code-block:: python

	test_set = original_model.generateSet(set_size=1000, param=10)

Now we can compute the loglikelihood under each model:

.. code-block:: python

	ll_original = original_model.logLikelihood(test_set)
	ll_output   =   output_model.logLikelihood(test_set)
	quality = ll_original - ll_output
	print(quality)

If ``quality`` is positive then we are overfitting.
