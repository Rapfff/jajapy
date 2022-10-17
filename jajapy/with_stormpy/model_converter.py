import stormpy as st
from numpy import zeros, where, full
from ..mc import MC
from ..mdp import MDP
from ..ctmc import CTMC

"""
def stormModelToJajapy(h):
	if type(h) == st.storage.storage.SparseMdp:
		ty = 2
	elif type(h) == st.storage.storage.SparseDtmc:
		ty = 0
	else:
		ty = 1

	labels = []
	for s in enumerate(h.states):
		for l in s.labels:
			labels.append(l)
	
	nb_states = len(labels)
	labels = list(set(labels))
	labels.remove("init")
	nb_labels = len(labels)
	
	if ty == 2:
		actions = []
		for s in h.states:
			for a in s.actions:
				actions.append('a'+str(a))
		actions = list(set(actions))
	
	matrix = [None for i in range(nb_states)]
	c = 1
	for s in h.states:
		for l in range(0,len(s.labels)-1):
			if ty == 2:
				print("ERROR: Cannot translate Storm MDPs with several labels for one state.")
				return False
			matrix[?] = [[0 for i in range(nb_labels)] for i in range(nb_states)]
			matrix[?][len(h.states)+c][labels.index(s.labels[l+1])]
			c += 1
	
		for a in s.actions:
			for t in a.transitions:
				dest = t.column
				obs = h.states[dest].labels[0]
				if ty == 2:
					matrix[?][int(str(a))][t.column][labels.index(obs)] = t.value()
				else:
					matrix[?][t.column][labels.index(obs)] = t.value()
	return None
"""
def jajapyModeltoStorm(h):
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
	state_index, c =  _buildStateIndex(h)
	transition_matrix, exit_rates = _buildTransitionMatrix(h,state_index,c)
	state_labeling = _buildStateLabeling(h,state_index,c)
	if type(h) == MDP:
		model = _buildMDP(h,c,transition_matrix,state_labeling)
	elif type(h) == CTMC:
		model = _buildCTMC(transition_matrix,state_labeling,exit_rates)
	else:
		model = _buildDTMC(transition_matrix,state_labeling)
	return model

def _buildMDP(h,c,transition_matrix,state_labeling):
	nb_actions = len(h.getActions())
	choice_labeling = st.storage.ChoiceLabeling(c*nb_actions)
	for a in h.getActions():
		choice_labeling.add_label(a)
	for ia,a in enumerate(h.getActions()):
		choice_labeling.add_label_to_choice(a,ia)

	reward_models = {}
	action_reward = [-1.0 for _ in range(len(transition_matrix))]
	reward_models["nb_executed_actions"] = st.SparseRewardModel(optional_state_action_reward_vector = action_reward)
	components = st.SparseModelComponents(transition_matrix=transition_matrix,
										  state_labeling=state_labeling,
										  reward_models=reward_models)
	components.choice_labeling = choice_labeling
	mdp = st.storage.SparseMdp(components)
	return mdp

def _buildCTMC(transition_matrix,state_labeling,exit_rates):
	components = st.SparseModelComponents(transition_matrix=transition_matrix,
										  state_labeling=state_labeling,
										  rate_transitions=True)
	components.exit_rates = exit_rates
	ctmc = st.storage.SparseCtmc(components)
	return ctmc

def _buildDTMC(transition_matrix, state_labeling):
	components = st.SparseModelComponents(transition_matrix=transition_matrix,
										  state_labeling=state_labeling)
	mc = st.storage.SparseDtmc(components)
	return mc

def _buildStateLabeling(h,state_index,c):
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
	return state_labeling

def _buildTransitionMatrix(h,state_index,c):
	if type(h) == MDP:
		nb_actions = len(h.getActions())
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
		return st.build_sparse_matrix(transition_matrix,[nb_actions*i for i in range(c)]), None
		
	elif type(h) == CTMC:
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
		return st.build_sparse_matrix(transition_matrix), exit_rates
	else:
		transition_matrix = zeros((c,c))
		for s in range(h.nb_states):
			p = zeros(c)
			if type(h) == MC:
				states, observations = where(h.matrix[s] > 0.0)
				for s2, o in zip(states,observations):
					p[state_index[s2][o]] = h.tau(s,s2,h.getAlphabet()[o])
			else:
				states = where(h.matrix[s] > 0.0)[0]
				observations = where(h.output[s] > 0.0)[0]
				for s2 in states:
					for o in observations:
						p[state_index[s2][o]] = h.tau(s,s2,h.getAlphabet()[o])
			for x in [i for i in state_index[s] if i > -1]:
				transition_matrix[x] = p
		return st.build_sparse_matrix(transition_matrix), None


def _buildStateIndex(h):
	if type(h) == MDP:
		state_index = full((h.nb_states, len(h.getAlphabet())+1),-1,dtype=int)
		c = 0

		for s in where(h.initial_state > 0.0)[0]:
			state_index[s][len(h.getAlphabet())] = c
			c += 1
		for s in range(h.nb_states):
			_, states, observations = where(h.matrix[s] > 0.0)
			for s2, o in zip(states,observations):
				if state_index[s2][o] == -1:
					state_index[s2][o] = c
					c += 1
	else:
		state_index = full((h.nb_states, len(h.getAlphabet())+1),-1,dtype=int)
		c = 0
		for s in where(h.initial_state > 0.0)[0]:
			state_index[s][len(h.getAlphabet())] = c
			c += 1
		for s in range(h.nb_states):
			if type(h) == MC or type(h) == CTMC:
				states, observations = where(h.matrix[s] > 0.0)
			else:
				states = where(h.matrix[s] > 0.0)[0]
				observations = where(h.output[s] > 0.0)[0]
			for s2 in states:
				for o in observations:
					if state_index[s2][o] == -1:
						state_index[s2][o] = c
						c += 1
	return state_index, c
