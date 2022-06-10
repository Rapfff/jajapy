References
==========

Models
------

`jajapy` supports different kind of Markov models that have different properties.

The following table summarizes the main properties of these models. The *first column* indicates
if, at each timestep, a model generates a discrete observation, a continuous observations, or a
vector of continuous observations. The *second column* shows if the observations are generated while
in a state or while moving from one state to another (in the first case the generation function is
independant to the transition function, not in the second case). The *third column* indicates
if the model is deterministic. And finally the *fourth column* shows if the model is a continuous
time model (or a discrete time model). A continuous time model will wait in each state for some period
of time before moving to another state.


======  ==================== ====================== ============= ===============
Model   Observations type    Observation generation Deterministic Continuous time
======  ==================== ====================== ============= ===============
HMM                 Discrete                  State           Yes              No
MC                  Discrete             Transition           Yes              No
MDP                 Discrete             Transition            No              No
CTMC                Discrete             Transition           Yes             Yes
GOHMM             Continuous                  State           Yes              No
MGOHMM  Vector of Continuous                  State           Yes              No
======  ==================== ====================== ============= ===============



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
   MM_CTMC_Composition

Alergia (state-merging) methods:

.. toctree::
   Alergia
   IOAlergia

Others
------
.. toctree::
	Set
	Tools