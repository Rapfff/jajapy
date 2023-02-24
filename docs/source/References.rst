References
==========

Models
------

*Jajapy* supports different kind of Markov models that have different properties.

The following table summarizes the main properties of these models. The *second column* indicates
if, at each timestep, a model generates a discrete observation, or a vector of continuous observations
(this vector can possibly contains only one value).
The *third column* indicates if the model is deterministic or not.
The *fourth* one shows if the model is a continuous time model (or a discrete time model).
A continuous time model will wait in each state for some period of time (called *dwell time*) before moving to another state.
Finally the last solumn indicates if the model is parametric.
In a parametric model, transition probabilities can be expressed are polynomial composition of parameters.
A parameter can also be involved in several transitions.

======  ==================== ============= =============== ==========
Model   Observations type    Deterministic Continuous time Parametric
======  ==================== ============= =============== ==========
HMM                 Discrete           Yes              No         No
MC                  Discrete           Yes              No         No
MDP                 Discrete            No              No         No
CTMC                Discrete           Yes             Yes         No
PCTMC               Discrete           Yes              No        Yes
GoHMM   Vector of Continuous           Yes              No         No
======  ==================== ============= =============== ==========

One can wander what is the difference between MC and HMM: each MC state is labelled with exactly one
observation, which is seen each time we are in this state. On the other hand, each HMM state is
associated with a probability distribution over the observations. Each time we are in this HMM state,
an observation is generated according to the probability distribution associated to this state.

.. toctree::
   
   HMM
   MC
   MDP
   CTMC
   PCTMC
   GoHMM


Learning Algorithms
-------------------
Classic Baum-Welch algorithms:

.. toctree::

   BW

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