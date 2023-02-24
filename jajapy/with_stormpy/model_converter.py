import stormpy as st
from numpy import zeros, array, newaxis, reshape, vstack, concatenate, hstack, newaxis, nan, full
from ..mc.MC import MC
from ..mdp.MDP import MDP
from ..ctmc.CTMC import CTMC
from ..pctmc.PCTMC import PCTMC
from copy import deepcopy
import os
from sympy import symbols, sympify

def stormpyModeltoJajapy(h,actions_name:list = [],from_prism=False):
	"""
	Given a stormpy.SparseCtmc, stormpy.SparseDtmc, stormpy.SparseMdp, or
	stormpy.SparseParametricCtmc, it returns the equivalent jajapy model.
	The output object will be a jajapy.MC, jajapy.CTMC, jajapy.MDP or
	jajapy.PCTMC depending on the input.

	Parameters
	----------
	h : stormpy.SparseCtmc, stormpy.SparseDtmc, stormpy.SparseMdp or stormpy.SparseParametricCtmc
		The model to convert.
	
	actions_name : list of str, optional.
		If the model is an MDP, the name of the actions in the output
		model will be the one in this list. Otherwise they will be
		`a0,a1,a2,...`.

	Returns
	-------
	jajapy.MC, jajapy.CTMC, jajapy.MDP or jajapy.PCTMC
		The same model in jajapy format.
	"""
	if type(h) == st.SparseDtmc:
		ty = 0
	elif type(h) == st.SparseCtmc:
		ty = 1
	elif type(h) == st.SparseMdp:
		ty = 2
	#elif type(h) == st.SparseParametricDtmc:
	#	ty = 3
	elif type(h) == st.SparseParametricCtmc:
		ty = 4
	else:
		raise TypeError(str(type(h))+' cannot be translated to Jajapy model.')
	
	labelling = [None for _ in range(len(h.states))]
	if ty == 2:
		actions = []
		for s in h.states:
			for a in s.actions:
				if len(actions_name) <= int(str(a)):
					actions.append('a'+str(a))
				else:
					actions.append(actions_name[int(str(a))])
		actions = list(set(actions))
		matrix = zeros((len(h.states),len(actions),len(h.states)))
	elif ty == 0 or ty == 1:
		matrix = zeros((len(h.states),len(h.states)))
	elif ty == 3 or ty == 4:
		matrix = zeros((len(h.states),len(h.states)),dtype='uint16')
		p_str = []
		p_v = {}
		p_i = []
		t_expr = [sympify(0.0)]

	add_init_state = None
	for si,s in enumerate(h.states):
		c = si
		temp = list(s.labels)
		if "deadlock" in temp:
			temp.remove("deadlock")
		temp.sort()
		if len(temp) == 0:
			labelling[si] = "empty"
		elif 'init' in temp and len(temp) > 1:
			temp.remove("init")
			labelling.append("init")
			labelling[si] = '_'.join(list(temp))
			add_init_state = c
		else:
			labelling[si] = '_'.join(list(temp))

		for a in s.actions:
			for t in a.transitions:
				dest = t.column
				t_val = t.value()
				if ty == 2:
					matrix[c][int(str(a))][dest] = t_val
				elif ty == 1 or ty == 0:
					matrix[c][dest] = t_val
				else:
					ps = [i.name for i in list(t_val.gather_variables())]
					if len(ps) == 1:
						ps = [symbols(ps[0])]
					elif len(ps) > 1:
						ps = list(symbols(" ".join(ps)))
					for v in ps:
						v = v.name
						if not v in p_str:
							p_str.append(v)
							p_i.append([])
							p_v[v] = nan
						p_i[p_str.index(v)].append([c,dest])
					t_val = sympify(str(t_val))
					if t_val.is_real or not t_val in t_expr:
						matrix[c][dest] = len(t_expr)
						t_expr.append(t_val)
					else:
						matrix[c][dest] = t_expr.index(t_val)
		if ty == 1 and not from_prism:
			matrix[c] *= h.exit_rates[si]
	
	if add_init_state != None:
		#matrix = vstack((matrix,matrix[add_init_state]))
		if ty == 2:
			matrix = vstack((matrix,zeros((1,matrix.shape[1],matrix.shape[2]))))
			matrix[-1].T[add_init_state] = full(matrix.shape[1],1.0)
			matrix = concatenate((matrix,zeros((matrix.shape[0],matrix.shape[1],1))),axis=2)
		elif ty == 1 or ty == 0:
			matrix = vstack((matrix,zeros((matrix.shape[0]))))
			matrix[-1][add_init_state] = 1.0
			matrix = hstack((matrix,zeros(len(matrix))[:,newaxis]))
		else:
			matrix = vstack((matrix,zeros((matrix.shape[0]),dtype='uint16')))
			t_val = sympify('1.0')
			matrix[-1][add_init_state] = len(t_expr)
			t_expr.append(t_val)
			matrix = hstack((matrix,zeros(len(matrix),dtype=('uint16'))[:,newaxis]))

	if ty == 0:
		return MC(matrix, labelling)
	elif ty == 1:
		return CTMC(matrix, labelling)
	elif ty == 2:
		return MDP(matrix,labelling,actions)
	#elif ty == 3:
	#	return PMC(matrix,labelling,p_v,p_i,p_str)
	elif ty == 4:
		return PCTMC(matrix,labelling,t_expr,p_v,p_i,p_str)

def jajapyModeltoStormpy(h):
	"""
	Given a jajapy.MC, a jajapy.CTMC, a jajapy.MDP or an instantiated
	jajapy.PCTMC, it returns the equivalent stormpy sparse model.
	The output object will be a stormpy.SparseCtmc, stormpy.SparseDtmc,
	stormpy.SparseMdp, or stormpy.SparseParametricCtmc depending on the input.

	Parameters
	----------
	h : jajapy.MC, jajapy.CTMC, jajapy.MDP or instantiated jajapy.PCTMC
		The model to convert.

	Returns
	-------
	stormpy.SparseCtmc, stormpy.SparseDtmc, stormpy.SparseMdp or stormpy.SparseParametricCtmc
		The same model in stormpy format.
	"""
	if type(h) == MDP:
		return MDPtoStormpy(h)
	elif type(h) == CTMC:
		return CTMCtoStormpy(h)
	elif type(h) == MC:
		return MCtoStormpy(h)
	elif type(h) == PCTMC:
		try:
			h = PCTMCtoCTMC(h)
		except ValueError:
			raise ValueError("Cannot convert non-instantiated PCTMC to Stormpy.")
		return CTMCtoStormpy(h)
	else:
		raise TypeError(str(type(h))+' cannot be translated to a stormpy sparse model.')

def _buildStateLabeling(h):
	state_labelling = st.storage.StateLabeling(h.nb_states)
	for o in h.getAlphabet():
		state_labelling.add_label(o)
	for s in range(h.nb_states):
		state_labelling.add_label_to_state(h.labelling[s],s)
	return state_labelling

def MDPtoStormpy(h):
	"""
	Given a jajapy.MDP, it returns the equivalent stormpy sparse model.
	The output object will be a stormpy.SparseMdp.

	Parameters
	----------
	h : jajapy.MDP
		The model to convert.

	Returns
	-------
	stormpy.SparseMdp
		The same model in stormpy format.
	"""
	state_labelling = _buildStateLabeling(h)
	nb_actions = len(h.getActions())
	transition_matrix = h.matrix
	transition_matrix = reshape(transition_matrix.flatten(),(h.nb_states*nb_actions,h.nb_states))
	transition_matrix =  st.build_sparse_matrix(transition_matrix,[nb_actions*i for i in range(h.nb_states)])
	choice_labelling = st.storage.ChoiceLabeling(h.nb_states*nb_actions)
	for ia,a in enumerate(h.getActions()):
		choice_labelling.add_label(a)
		choice_labelling.add_label_to_choice(a,ia)
	reward_models = {}
	action_reward = [-1.0 for _ in range(len(transition_matrix))]
	reward_models["nb_executed_actions"] = st.SparseRewardModel(optional_state_action_reward_vector = action_reward)
	components = st.SparseModelComponents(transition_matrix=transition_matrix,
										  state_labeling=state_labelling,
										  reward_models=reward_models)
	components.choice_labeling = choice_labelling
	mdp = st.storage.SparseMdp(components)
	return mdp

def MCtoStormpy(h):
	"""
	Given a jajapy.MC, it returns the equivalent stormpy sparse model.
	The output object will be a stormpy.SparseDtmc.

	Parameters
	----------
	h : jajapy.MC
		The model to convert.

	Returns
	-------
	stormpy.SparseDtmc
		The same model in stormpy format.
	"""
	state_labelling = _buildStateLabeling(h)
	transition_matrix = h.matrix
	transition_matrix =  st.build_sparse_matrix(transition_matrix)
	components = st.SparseModelComponents(transition_matrix=transition_matrix,
										  state_labeling=state_labelling)
	mc = st.storage.SparseDtmc(components)
	return mc

def CTMCtoStormpy(h):
	"""
	Given a jajapy.CTMC, it returns the equivalent stormpy sparse model.
	The output object will be a stormpy.SparseCtmc.

	Parameters
	----------
	h : jajapy.CTMC
		The model to convert.

	Returns
	-------
	stormpy.SparseCtmc
		The same model in stormpy format.
	"""
	state_labelling = _buildStateLabeling(h)
	transition_matrix = deepcopy(h.matrix)
	e = array([h.e(s) for s in range(h.nb_states)])
	transition_matrix /= e[:,newaxis]
	transition_matrix =  st.build_sparse_matrix(transition_matrix)
	components = st.SparseModelComponents(transition_matrix=transition_matrix,
										  state_labeling=state_labelling,
										  rate_transitions=True)
	components.exit_rates = e
	ctmc = st.storage.SparseCtmc(components)
	return ctmc

def PCTMCtoCTMC(h: PCTMC) -> CTMC:
	"""
	Translates a given instantiated PCTMC to an equivalent CTMC.

	Parameters
	----------
	h : PCTMC
		An instantiated PCTMC.

	Returns
	-------
	CTMC
		The equivalent CTMC.

	Raises
	------
	ValueError
		If `h` is a non-instantiated PCTMC.
	"""
	if not h.isInstantiated():
		raise ValueError("Cannot convert non-instantiated PCTMC to CTMC.")
	res = zeros(h.matrix.shape)
	for s in range(h.nb_states):
		for ss in range(h.nb_states):
			res[s,ss] = h.transitionValue(s,ss)
	return CTMC(res, h.labelling, h.name)

def loadPrism(path: str):
	"""
	Load the model described in file `path` under Prism format.
	Remark: this function uses the stormpy parser for Prism file.

	Remarks
	-------
	For technical reason, this function clear the terminal on usage.

	Parameters
	----------
	path : str
		Path to the Prism model to load.

	Returns
	-------
	jajapy.MC, jajapy.CTMC, jajapy.MDP or jajapy.PCTMC
		A jajapy model equivalent to the model described in `path`.
	"""
	try:
		prism_program = st.parse_prism_program(path,False)
	except RuntimeError:
		prism_program = st.parse_prism_program(path,True)
	try:
		stormpy_model = st.build_model(prism_program)
	except RuntimeError:
		stormpy_model = st.build_parametric_model(prism_program)

	if os.name != "nt":
		os.system('clear')
	else:
		os.system('cls')

	jajapy_model = stormpyModeltoJajapy(stormpy_model,from_prism=True)
	return jajapy_model