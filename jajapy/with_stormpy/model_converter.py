import stormpy as st
from numpy import zeros, where, full, array, delete
from ..mc import MC
from ..mdp import MDP
from ..ctmc import CTMC

def stormModeltoJajapy(h,actions_name:list = []):

	def mergeStates(m,i,j):
		if type(m) != MDP:
			for s in range(m.nb_states):
				tmp = array([m.matrix[s][x] if (x!=i and x != j) else m.matrix[s][i]+m.matrix[s][j] for x in range(m.nb_states)])
				m.matrix[s] = tmp
			m.matrix = delete(m.matrix,j,0)
			m.matrix = delete(m.matrix,j,1)
		else:
			for s in range(m.nb_states):
				tmp = zeros((len(m.actions),m.nb_states,len(m.alphabet)))
				for a in range(len(m.actions)):
					tmp[a] = array([m.matrix[s][a][x] if (x!=i and x != j) else m.matrix[s][a][i]+m.matrix[s][a][j] for x in range(m.nb_states)])
				m.matrix[s] = tmp
			m.matrix = delete(m.matrix,j,0)
			m.matrix = delete(m.matrix,j,2)

		m.initial_state[i] += m.initial_state[j]
		m.initial_state = delete(m.initial_state,j)
		m.nb_states -= 1

	def reduced(m):
		#remove unused observation
		if type(m) != MDP:
			while len(where(m.matrix.sum(axis=1).sum(axis=0) == 0)[0]) > 0:
				to_delete = where(m.matrix.sum(axis=1).sum(axis=0) == 0)[0][0]
				m.matrix = delete(m.matrix,to_delete,2)
				m.alphabet.remove(m.alphabet[to_delete])
		else:
			while len(where(m.matrix.sum(axis=1).sum(axis=1).sum(axis=0) == 0)[0]) > 0:
				to_delete = where(m.matrix.sum(axis=1).sum(axis=1).sum(axis=0) == 0)[0][0]
				m.matrix = delete(m.matrix,to_delete,3)
				m.alphabet.remove(m.alphabet[to_delete])

		i = 0
		while i < m.nb_states-1:
			j = i+1
			while j < m.nb_states:
				if type(m) != MDP:
					tmp_i = array([m.matrix[i][x] if (x!=i and x != j) else m.matrix[i][i]+m.matrix[i][j] for x in range(m.nb_states)])
					tmp_i = delete(tmp_i,j)

					tmp_j = array([m.matrix[j][x] if (x!=i and x != j) else m.matrix[j][i]+m.matrix[j][j] for x in range(m.nb_states)])
					tmp_j = delete(tmp_j,j)
				else:
					tmp_i = zeros((len(m.actions),m.nb_states-1,len(m.alphabet)))
					tmp_j = zeros((len(m.actions),m.nb_states-1,len(m.alphabet)))
					for a in range(len(m.actions)):
						t = array([m.matrix[i][a][x] if (x!=i and x != j) else m.matrix[i][a][i]+m.matrix[i][a][j] for x in range(m.nb_states)])
						t = delete(t,j,0)
						tmp_i[a] = t
						t = array([m.matrix[j][a][x] if (x!=i and x != j) else m.matrix[j][a][i]+m.matrix[j][a][j] for x in range(m.nb_states)])
						t = delete(t,j,0)
						tmp_j[a] = t

				if (tmp_i==tmp_j).all():
					mergeStates(m,i,j)
					j -= 1
				j += 1
			i += 1


	if type(h) == st.SparseDtmc:
		ty = 0
	elif type(h) == st.SparseCtmc:
		ty = 1
	else:
		ty = 2

	labels = []
	state_labels = []
	actions = [] # for MDPs
	init = []

	for s in h.states:		
		state_labels.append(list(s.labels))
		labels += list(s.labels)

		if 'init' in s.labels:
			init.append(int(str(s)))
			state_labels[-1].remove('init')

		if len(state_labels[-1]) == 0:
			labels.append("empty")
			state_labels[-1].append('empty')

		if ty > 0:
			if len(state_labels[-1]) > 1:
				print("ERROR") # for MDPs => empty action, for CTMCs => no time
				return False
			for a in s.actions:
				if len(actions_name) <= int(str(a)):
					actions.append('a'+str(a))
				else:
					actions.append(actions_name[int(str(a))])

	labels = list(set(labels))
	labels.remove('init')
	actions = list(set(actions))
	c = 0 
	states_id_mapping = [0]
	for i in range(1, len(state_labels)):
		c += len(state_labels[i-1])
		states_id_mapping.append(c)

	nb_states = states_id_mapping[-1] + len(state_labels[-1])
	nb_labels = len(labels)
	nb_actions = len(actions)

	if ty < 2:
		matrix = zeros((nb_states,nb_states,nb_labels))
	else:
		matrix = zeros((nb_states,nb_actions,nb_states,nb_labels))

	c = 0
	for s in range(len(state_labels)):
		for l in state_labels[s][1:]:
			matrix[c][c+1][labels.index(l)] = 1.0
			c += 1
		
		for a in h.states[s].actions:
			for t in a.transitions:
				dest = states_id_mapping[t.column]
				l = labels.index(state_labels[t.column][0])
				if ty < 2:
					matrix[c][dest][l] = t.value()
				else:
					matrix[c][int(str(a))][dest][l] = t.value()
		c += 1

	initial_states = [0.0 for i in range(c)]
	for i in init:
		initial_states[states_id_mapping[i]] = 1.0/len(init)

	if ty == 0:
		m = MC(matrix, labels, initial_states)
	elif ty == 1:
		for s in range(len(matrix)):
			matrix[s] *= h.exit_rates[s]
		m = CTMC(matrix, labels, initial_states)
	else:
		m = MDP(matrix, labels, actions, initial_states)
	
	reduced(m)
	return m


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
