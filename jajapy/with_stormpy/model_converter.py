import stormpy as st
from numpy import zeros, array, insert, where, full
from ..hmm import HMM
from ..mc import MC
from ..mdp import MDP
from ..ctmc import CTMC

def modeltoStorm(h):
	"""
	Returns a trace-equivalent model that can be use by Storm.
	The input model can be a jajapy.HMM, a jajapy.MC, a jajapy.CTMC or
	a jajapy.MDP.
	The output object will be a stormpy.SparseCtmc, stormpy.SparseDtmc,
	or a stormpy.SparseMdp depending on the input.

	Parameters
	----------
	h : jajapy.HMM, jajapy.MC, jajapy.CTMC or jajapy.MDP
		The model to convert.

	Returns
	-------
	stormpy.SparseCtmc, stormpy.SparseDtmc or stormpy.SparseMdp
		A trace-equivalent model.
	"""
	if type(h) != MDP and type(h) != CTMC and type(h) != MC and type(h) != HMM:
		print("The input model should be a MDP, a CTMC, a MC or a HMM.")
		return None
	if type(h) == MDP:
		return _MDPtoStorm(h)
	if type(h) == CTMC:
		return _CTMCtoStorm(h)

	nb_states = array([len(h.getAlphabet(i)) for i in range(h.nb_states)])
	nb_states_i = nb_states.cumsum()
	nb_states_i = insert(nb_states_i,0,0)
	nb_states = sum(nb_states)
	transition_matrix = zeros((nb_states+1,nb_states+1))

	for sh in where(h.initial_state > 0.0)[0]:
		for i,o in enumerate(h.getAlphabet(sh)):
			ss = nb_states_i[sh]
			transition_matrix[nb_states][ss+i] = h.pi(sh)*h.b(sh,o)

	if type(h) == HMM:
		_HMMtoStorm(h,transition_matrix,nb_states_i)
	elif type(h) == MC or type(h) == CTMC:
		_MCtoStorm( h,transition_matrix,nb_states_i)

	transition_matrix = st.build_sparse_matrix(transition_matrix)

	state_labeling = st.storage.StateLabeling(nb_states+1)
	state_labeling.add_label('init')
	for o in h.getAlphabet():
		state_labeling.add_label(o)
	
	for s in range(h.nb_states):
		for i,o in enumerate(h.getAlphabet(s)):
			state_labeling.add_label_to_state(o,nb_states_i[s]+i)
	state_labeling.add_label_to_state('init',nb_states)
	
	components = st.SparseModelComponents(transition_matrix, state_labeling)

	return st.storage.SparseDtmc(components)


def _MDPtoStorm(h):
	state_index = full((h.nb_states, len(h.getAlphabet())+1),-1,dtype=int)
	nb_actions = len(h.getActions())
	c = 0

	for s in where(h.initial_state > 0.0)[0]:
		state_index[s][len(h.getAlphabet())] = c
		c += 1
	for s in range(h.nb_states):
		actions, states, observations = where(h.matrix[s] > 0.0)
		for s2, o in zip(states,observations):
			if state_index[s2][o] == -1:
				state_index[s2][o] = c
				c += 1

	transition_matrix = zeros((c*nb_actions,c))
	
	for s in range(h.nb_states):
		for a in h.getActions(s):
			p = zeros(c)
			ia = h.actions.index(a)
			states, observations = where(h.matrix[s][ia] > 0.0)
			for s2, o in zip(states,observations):
				p[state_index[s2][o]] = h.tau(s,a,s2,h.getAlphabet()[o])
			for x in [i for i in state_index[s] if i > -1]:
				transition_matrix[x*nb_actions+ia] = p
	transition_matrix = st.build_sparse_matrix(transition_matrix,[nb_actions*i for i in range(c)])

	state_labeling = st.storage.StateLabeling(c)
	for o in h.getAlphabet():
		state_labeling.add_label(o)
	state_labeling.add_label('init')
	states, observations = where(state_index > -1)
	for s,o in zip(states,observations):
		if o == len(h.getAlphabet()):
			state_labeling.add_label_to_state('init',state_index[s][o])
		else:
			state_labeling.add_label_to_state(h.getAlphabet()[o],state_index[s][o])

	choice_labeling = st.storage.ChoiceLabeling(c*nb_actions)
	for a in h.getActions():
		choice_labeling.add_label(a)
	for ia,a in enumerate(h.getActions()):
		choice_labeling.add_label_to_choice(a,ia)
	components = st.SparseModelComponents(transition_matrix=transition_matrix,
										  state_labeling=state_labeling)
	components.choice_labeling = choice_labeling
	mdp = st.storage.SparseMdp(components)
	return mdp


def _CTMCtoStorm(h):
	state_index = full((h.nb_states, len(h.getAlphabet())+1),-1,dtype=int)
	c = 0
	for s in where(h.initial_state > 0.0)[0]:
		state_index[s][len(h.getAlphabet())] = c
		c += 1
	for s in range(h.nb_states):
		states, observations = where(h.matrix[s] > 0.0)
		for s2, o in zip(states,observations):
			if state_index[s2][o] == -1:
				state_index[s2][o] = c
				c += 1

	transition_matrix = zeros((c,c))
	exit_rates = zeros(c)
	
	for s in range(h.nb_states):
		p = zeros(c)
		states, observations = where(h.matrix[s] > 0.0)
		for s2, o in zip(states,observations):
			p[state_index[s2][o]] = h.tau(s,s2,h.getAlphabet()[o])
		for x in [i for i in state_index[s] if i > -1]:
			transition_matrix[x] = p
			exit_rates[x] = h.e(s)
	transition_matrix = st.build_sparse_matrix(transition_matrix)

	
	state_labeling = st.storage.StateLabeling(c)
	for o in h.getAlphabet():
		state_labeling.add_label(o)
	state_labeling.add_label('init')
	states, observations = where(state_index > -1)
	for s,o in zip(states,observations):
		if o == len(h.getAlphabet()):
			state_labeling.add_label_to_state('init',state_index[s][o])
		else:
			state_labeling.add_label_to_state(h.getAlphabet()[o],state_index[s][o])

	components = st.SparseModelComponents(transition_matrix=transition_matrix,
										  state_labeling=state_labeling,
										  rate_transitions=True)
	components.exit_rates = exit_rates
	ctmc = st.storage.SparseCtmc(components)
	return ctmc
				

def _HMMtoStorm(h,transition_matrix,nb_states_i):
	for s1h in range(h.nb_states):
		for i1 in range(len(h.getAlphabet(s1h))):
			for s2h in where(h.matrix[s1h] > 0.0)[0]:
				for i2,o2 in enumerate(h.getAlphabet(s2h)):
					src = nb_states_i[s1h]+i1
					dst = nb_states_i[s2h]+i2
					transition_matrix[src][dst] = h.a(s1h,s2h)*h.b(s2h,o2)

def _MCtoStorm(h,transition_matrix,nb_states_i):
	for s1h in range(h.nb_states):
		for i1,o1 in enumerate(h.getAlphabet(s1h)):
			for s2h in where(h.matrix[s1h].T[h.alphabet.index(o1)] > 0.0)[0]:
				for i2,o2 in enumerate(h.getAlphabet(s2h)):
					src = nb_states_i[s1h]+i1
					dst = nb_states_i[s2h]+i2
					transition_matrix[src][dst] = h.b(s2h,o2)*h.tau(s1h,s2h,o1)/h.b(s1h,o1)
	