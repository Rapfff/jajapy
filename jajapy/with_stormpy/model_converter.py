import stormpy as st
from numpy import zeros, array, insert, where

def HMMtoDTMC(h):
	nb_states = array([len(h.getAlphabet(i)) for i in range(h.nb_states)])
	nb_states_i = nb_states.cumsum()
	nb_states_i = insert(nb_states_i,0,0)
	nb_states = sum(nb_states)
	transition_matrix = zeros((nb_states+1,nb_states+1))

	for s1h in range(h.nb_states):
		for s2h in range(h.nb_states):
			if h.a(s1h,s2h) > 0.0:
				s2s = nb_states_i[s2h]
				for i,o in enumerate(h.getAlphabet(s2h)):
					for s1s in range(nb_states_i[s1h],nb_states_i[s1h+1]):
						transition_matrix[s1s][s2s+i] = h.a(s1h,s2h)*h.b(s2h,o)

	for sh in where(h.initial_state > 0.0)[0]:
		for i,o in enumerate(h.getAlphabet(sh)):
			ss = nb_states_i[sh]
			transition_matrix[nb_states][ss+i] = h.pi(sh)*h.b(sh,o)
	
	print(transition_matrix)
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
	dtmc = st.storage.SparseDtmc(components)
	return dtmc

