References
==========

Models
------

`jajapy` supports different kind of Markov models that have different properties.

The following table summarizes the main properties of these models. The *first column* indicates
if, at each timestep, a model generates a discrete observation, a continuous observations, or a
vector of continuous observations. The *second column* indicates if the model is deterministic or not.
And finally the *third column* shows if the model is a continuous time model (or a discrete time model).
A continuous time model will wait in each state for some period of time before moving to another state.

======  ==================== ============= ===============
Model   Observations type    Deterministic Continuous time
======  ==================== ============= ===============
HMM                 Discrete           Yes              No
MC                  Discrete           Yes              No
MDP                 Discrete            No              No
CTMC                Discrete           Yes             Yes
GOHMM             Continuous           Yes              No
MGOHMM  Vector of Continuous           Yes              No
======  ==================== ============= ===============

One can wander what is the difference between MC and HMM: each MC state is labelled with exactly one
observation, which is seen each time we are in this state. On the other hand, each HMM state is
associated with a probability distribution over the observations. Each time we are in this HMM state,
an observation is generated according to the probability distribution associated to this state.

.. toctree::
   :maxdepth: 1
   
   HMM
   MC
   MDP
   CTMC
   GOHMM
   MGOHMM


Learning Algorithms
-------------------
Classic Baum-Welch algorithms:

.. toctree::
   BW_HMM
   BW_MC
   BW_GOHMM
   BW_MGOHMM
   BW_MDP
   BW_CTMC


Advanced extensions:

.. toctree::
   Active_BW_MDP

Alergia (state-merging) methods:

.. toctree::
   Alergia
   IOAlergia

Others
------
.. toctree::
	Set
	Tools
	working_with_stormpy