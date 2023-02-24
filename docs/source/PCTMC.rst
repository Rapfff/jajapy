Parametric Continuous Time Markov Chain (PCTMC)
===============================================
A PCTMC is a CTMC where the transition rates can be expressed as polynomial
functions over a set of parameters.

Example
-------

.. image:: pictures/Tandem.png
	:width: 60%
	:align: center

The model above represents the `tandem queueing network <http://www.prismmodelchecker.org/casestudies/tandem.php>`_
where *c=3*. The yellow model corresponds to the *serverC*, and the green one the *SeverM*.
For each yellow state, the first value corresponds to the value of *sc* and the second one the value of *phi*.
The red transitions are the synchronous transitions (the *route* transitions in the Prism file), while the black
transitions are the asynchronous one.

Creation
^^^^^^^^
We can load the Prism file describing a PCTMC or a composition of PCMTCs as follow:

.. code-block:: python

	>>> import jajapy as ja
	>>> model = ja.loadPrism("tandem_3.sm")

We can also create it manually using the ``createPCTMC`` function (the following will create a model for *serverC* only):

.. code-block:: python

	>>> labelling = ['sc_0_ph_1','sc_0_ph_2',
	>>> 		'sc_1_ph_1','sc_1_ph_2',
	>>> 		'sc_2_ph_1','sc_2_ph_2',
	>>> 		'sc_3_ph_1','sc_3_ph_2',]
	>>> transitions = [(0,2,'lambda'),(2,4,'lambda'),(4,6,'lambda'),
	>>> 		(1,3,'lambda'),(3,5,'lambda'),(5,7,'lambda'),
	>>> 		(2,3,'mu1a'),(4,5,'mu1a'),(6,7,'mu1a')]
	>>> sync_trans = [(2,'route',0,'mu1b'),(4,'route',2,'mu1b'),(6,'route',4,'mu1b'),
	>>> 		(3,'route',0,'mu2'),(5,'route',2,'mu2'),(7,'route',4,'mu2')]
	>>> parameters = ['lambda','mu1a','mu1b','mu2']
	>>> initial_state = 0
	>>> parameter_instantiation = {'lambda':6.0}
	>>> model = createPCTMC(transitions,labelling,parameters,initial_state,parameter_instantiation,sync_trans)

It is not possible to create a random PCTMC. However, we can ask *Jajapy* to randomly
instantiate the parameters:

.. code-block:: python

	>>> model.isInstantiated()
	False
	>>> model.randomInstantiation(min_val=0.5, max_val=5.0)
	>>> model.isInstantiated()
	True

Converting from/to Stormpy
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	>>> stormpy_sparse_model = model.toStormpy() # the next line is equivalent
	>>> stormpy_sparse_model = ja.jajapyModeltoStormpy(model)
	>>> same_model == ja.stormpyModeltoJajapy(stormpy_sparse_model) 

Converting from/to Prism
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	>>> model.savePrism("my_ctmc.sm")
	>>> same_model = ja.loadPrism("my_ctmc.sm")

Synchronous composition
^^^^^^^^^^^^^^^^^^^^^^^
Two PCTMCs (or more) can be composed to create one PCTMC.
PCTMCs composition works as CTMCs composition.

.. code-block:: python

	>>> composition = synchronousCompositionPCTMCs([pctmc1, pctmc2, pctmc3])

Model
-----

.. autoclass:: jajapy.PCTMC
   :members:
   :inherited-members:

Other Functions
---------------
.. autofunction:: jajapy.createPCTMC

.. autofunction:: jajapy.loadPCTMC

.. autofunction:: jajapy.synchronousCompositionPCTMCs