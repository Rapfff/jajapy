from ..base.tools import resolveRandom
from ..base.Parametric_Model import *
from ..base.Set import Set
from numpy import ndarray, array, zeros, dot, nan, newaxis, hstack, inf, vstack, delete
from numpy.random import exponential, rand
from math import exp, log
from multiprocessing import cpu_count, Pool
from sys import platform

class PCTMC(Parametric_Model):
	"""
	Class representing a PCTMC.
	"""
	def __init__(self, matrix: ndarray, labeling: list,
				 transition_str: list, parameter_values: ndarray,
				 parameter_indexes: list, parameter_str: list,
				 name: str="unknow_PCTMC",
				 synchronous_transitions: list = []) -> None:
		"""
		Creates an PCTMC.

		Parameters
		----------
		matrix : ndarray
			Represents the transition matrix.
			matrix[i,j] is the index, in `transition_str`, of the symbolic
			value of the transition from `i` to `j`.
		labeling: list of str
			A list of N observations (with N the nb of states).
			If `labeling[s] == o` then state of ID `s` is labelled by `o`.
			Each state has exactly one label.
		transition_str: list of str
			Contains the symbolic value for each transition.
		parameter_values: list of float
			Contains the value for each parameter.
			`parameter_values[i]` is the instantiation for parameter `i`.
			If the ith parameter is not instantiated, `parameter_values[i] == nan`.
		parameter_indexes: list of ndarray
			Contains the indexes of each transition using each parameter.
			`parameter_indexes[i] = array([[0,1],[2,1]])` means that parameter `i`
			is used by the transition from state 0 to state 1 and from state 2 to state 1.
		parameter_str: list
			Contains the name of each parameter.
			`parameter_str[i]` is the name of parameter `i`.
			Parameter `i` doesn't have a name if `parameter_str[i]==None`.
		name : str, optional
			Name of the model.
			Default is "unknow_PCTMC"
		synchronous_transitions: list, optional.
			This is useful only for synchronously composing this PCTMC with
			another one.
			List of (source_state <int>, action <str>, dest_state <int>, rate <float>).
			Default is an empty list.
		"""
		self.synchronous_transitions = synchronous_transitions
		super().__init__(matrix,labeling,transition_str,parameter_values,parameter_indexes,parameter_str,name)

	def e(self,s: int) -> float:
		"""
		Returns the exit rate of state ``s``, i.e. the sum of all the rates in
		this state.

		Returns
		-------
		s : int
			A state ID.
		float
			An exit rate.
		"""
		if self.isInstantiated(s):
			return sum([self.evaluateTransition(s,i) for i in range(self.nb_states)])
		print("WARN: cannot compute the exit rate of non-instantiated state "+str(s))

	def l(self, s1:int, s2:int, obs:str) -> float:
		"""
		Returns the rate associated to the transition from state `s1`, seeing
		`obs`, to state `s2`.

		Parameters
		----------
		s1 : int
			A state ID.
		s2 : int
			A state ID.
		obs : str
			An observation.

		Returns
		-------
		float
			A rate.
		"""
		self._checkStateIndex(s1)
		self._checkStateIndex(s2)
		if self.labeling[s1] != obs:
			return 0.0
		return self.evaluateTransition(s1,s2)
	
	def lkl(self, s: int, t: float) -> float:
		"""
		Returns the likelihood of staying in `s` state for a duration `t`.

		Parameters
		----------
		s : int
			A state ID.
		t : float
			A waiting time.

		Returns
		-------
		float
			A Likelihood.
		"""
		if t < 0.0:
			return 0.0
		return self.e(s)*exp(-self.e(s)*t)
	
	def tau(self, s1:int, s2: int, obs: str) -> float:
		"""
		Returns the probability to move from `s1` to `s2` generating
		observation ``obs``.

		Parameters
		----------
		s1 : int
			A state ID.
		s2 : int
			A state ID.
		obs : str
			An observation.

		Returns
		-------
		float
			A probability.
		"""
		e = self.e(s1)
		if e == 0.0:
			return inf
		return self.l(s1,s2, obs)/e
	
	def expected_time(self, s:int) -> float:
		"""
		Returns the expected waiting time in `s`, i.e. the inverse of
		the exit rate.
		
		Parameters
		----------
		s : int
			A state ID.
		
		Returns
		-------
		float
			expected waiting time in this state.
		"""
		
		e = self.e(s)
		if e == 0.0:
			return inf
		return 1/e
	
	def _checkRates(self,values=None) -> bool:
		if type(values) == type(None):
			values = self.parameter_values
		for p in self.parameter_values:
			if not isnan(p):
				if p < 0.0:
					return False
		return True
	
	def instantiate(self, parameters: list, values: list) -> bool:
		"""
		Set all the parameters in `parameters` to the values `values`.

		Parameters
		----------
		parameters : list of string
			List of all parameters to set. This list must contain parameter
			names.
		values : list of float
			List of values. `parameters[i]` will be set to `values[i]`.
		"""
		new_values =  super().instantiate(parameters, values)
		params_id = set([self.parameter_str(i) for i in parameters])
		for i in range(self.nb_states):
			for j in range(self.nb_states):
				if params_id & set(self.involvedParameters(i,j)):
					if self.evaluateTransition(i,j,new_values) < 0.0:
						print("WARN: invalid values. Instantiation ignored.")
						return False
		self.parameter_values = new_values
		self._evaluateAll(params_id)
		return True

	def randomInstantiation(self, parameters: list,min_val:float,max_val:float) -> None:
		val = rand((max_val-min_val)*len(parameters)+min_val)
		while not self.instantiate(parameters,val):
			val = rand((max_val-min_val)*len(parameters)+min_val)
	
	def _stateToString(self,state:int) -> str:
		res = "----STATE "+str(state)+"--"+self.labeling[state]+"----\n"
		if self.isInstantiated(state):
			res += "Exepected waiting time: "+str(self.expected_time(state))+'\n'
		for j in range(len(self.matrix[state])):
			if self.matrix[state][j] != 0:
				val = self.transitionStr(state,j)
				try:
					float(val)
				except ValueError:
					if not isnan(self.transitionValue(state,j)):
						val+=' (='+str(self.transitionValue(state,j))+')'
				res += "s"+str(state)+" -> s"+str(j)+" : lambda = "+val+'\n'
		return res
	
	def next(self,state: int) -> tuple:
		"""
		Return a state-observation pair according to the distributions 
		described by matrix
		
		Returns
		-------
		output : (int, str)
			A state-observation pair.
		"""
		if not self.isInstantiated(state):
			raise ValueError("At least one of the parameter is not instantiated.")
		exps = []
		for exp_lambda in [self.transitionValue(state,j) for j in range(self.nb_states)]:
			if exp_lambda == 0.0:
				exps.append(inf)
			else:
				exps.append(exponential(1/exp_lambda))
		next_state= exps.index(min(exps))
		next_obs = self.labeling[state]
		return (next_obs, next_state, min(exps))

	def run(self,number_steps: int, timed: bool = False) -> list:
		"""
		Simulates a run of length ``number_steps`` of the model and return the
		sequence of observations generated. If ``timed`` it returns a list of
		pairs waiting time-observation.
		
		Parameters
		----------
		number_steps: int
			length of the simulation.
		timed: bool, optional
			Wether or not it returns also the waiting times. Default is False.

		Returns
		-------
		output: list of str
			trace generated by the run.
		"""
		output = []
		current = resolveRandom(self.initial_state)
		c = 0
		while c < number_steps:
			[symbol, next_state, time_spent] = self.next(current)
			output.append(symbol)
			if timed:
				output.append(time_spent)
			current = next_state
			c += 1
		output.append(self.labeling[current])
		return output
	
	def _computeAlphas_timed(self, sequence: list, times: int) -> float:
		obs_seq   = [sequence[i] for i in range(0,len(sequence),2)]
		times_seq = [sequence[i] for i in range(1,len(sequence),2)]
		len_seq = len(obs_seq)-1
		prev_arr = array(self.initial_state)
		for k in range(len_seq):
			new_arr = zeros(self.nb_states)
			for s in range(self.nb_states):
				p = array([self.l(ss,s,obs_seq[k])*exp(-self.e(ss)*times_seq[k]) for ss in range(self.nb_states)])
				new_arr[s] = dot(prev_arr,p)
			prev_arr = new_arr
		last_arr = prev_arr * (array(self.labeling) == obs_seq[-1])
		return log(last_arr.sum())*times

	def logLikelihood(self,traces: Set) -> float:
		if traces.type == 0: # non-timed traces
			return super().logLikelihood(traces)
		else: # timed traces
			if platform != "win32" and platform != "darwin":
				p = Pool(processes = cpu_count()-1)
				tasks = []
				for seq,times in zip(traces.sequences,traces.times):
					tasks.append(p.apply_async(self._computeAlphas_timed, [seq, times,]))
				temp = [res.get() for res in tasks if res.get() != False]
			else:
				temp = [self._computeAlphas_timed(seq,times) for seq,times in zip(traces.sequences,traces.times)]
			return sum(temp)/sum(traces.times)

	def save(self,file_path:str) -> None:
		"""Save the model into a text file.

		Parameters
		----------
		file_path : str
			path of the output file.
		
		Examples
		--------
		>>> model.save("my_model.txt")
		"""
		f = open(file_path, 'w')
		f.write("PCTMC\n")
		super()._save(f)
	
	#def toCTMC(self,name='unknow_CTMC') -> CTMC:
	#	if not self._isInstantiated():
	#		raise ValueError("The model must be instantiated before being translated to a CTMC.")
	#	matrix = zeros(len(self.matrix),len(self.matrix))
	#	for i in self.matrix:
	#		for j in self.matrix:
	#			matrix[i,j] = self.transitionValue(i,j)
	#	return CTMC(matrix, self.labeling, name)

def loadPCTMC(file_path: str) -> PCTMC:
	"""
	Load an PCTMC saved into a text file.

	Parameters
	----------
	file_path : str
		Location of the text file.
	
	Returns
	-------
	output : PCTMC
		The PCTMC saved in `file_path`.
	
	Examples
	--------
	>>> model = loadPCTMC("my_model.txt")
	"""
	f = open(file_path,'r')
	l = f.readline()[:-1] 
	if l != " PCTMC":
		msg = "This file doesn't describe a PCTMC: it describes a "+l
		raise ValueError(msg)
	matrix,labeling,parameter_values,parameter_indexes,parameter_str,transition_str,name = loadParametricModel(f)
	f.close()
	return PCTMC(matrix,labeling,transition_str,parameter_values,
				 parameter_indexes,parameter_str,name)

def createPCTMC(transitions: list, labeling: list, initial_state,
				parameter_instantiation: dict={}, synchronous_transitions=[],
				name: str ="unknown_PCTMC") -> PCTMC:
	"""
	An user-friendly way to create a PCTMC.

	Parameters
	----------
	transitions : [ list of tuples (int, int, float or str)]
		Each tuple represents a transition as follow: 
		(source state ID, destination state ID, probability).
		The probability can be explicitly given (then it's a float),
		or a parameter (then it's the name of the parameter).
	labeling: list of str
		A list of N observations (with N the nb of states).
		If `labeling[s] == o` then state of ID `s` is labelled by `o`.
		Each state has exactly one label.
	parameter_instantiation: dict
		An instantion for some (or all) parameters.
		`parameter_instantiation == {'p':0.5}` means that parameter `p`
		should be instantiated to 0.5. The other parameters are not
		instantiated.
	initial_state : int or list of float
		Determine which state is the initial one (then it's the id of the
		state), or what are the probability to start in each state (then it's
		a list of probabilities).
	synchronous_transitions: list, optional.
		This is useful only for synchronously composing this PCTMC with
		another one.
		List of (source_state <int>, action <str>, dest_state <int>, rate <float>).
		Default is an empty list.
	name : str, optional
		Name of the model.
		Default is "unknow_PCTMC"
	
	Returns
	-------
	PCTMC
		the PCTMC describes by `transitions`, `labeling`, and `initial_state`.
	"""
	if 'init' in labeling:
		msg =  "The label 'init' cannot be used: it is reserved for initial states."
		raise SyntaxError(msg)
	
	labeling.append('init')
	
	states = [i[0] for i in transitions]+[i[1] for i in transitions]
	states += [i[0] for i in synchronous_transitions]+[i[2] for i in synchronous_transitions]
	states = list(set(states))
	states.sort()
	nb_states = len(states)
	if type(initial_state) == int:
		transitions.append((nb_states,initial_state,1.0))
	else:
		for i,j in enumerate(initial_state):
			transitions.append((nb_states,i,j))
	nb_states += 1
	if nb_states > len(labeling):
		raise ValueError("All states are not labelled (the labeling list is too small).")
	elif nb_states < len(labeling):
		print("WARNING: the labeling list is bigger than the number of states")

	parameter_str = []
	parameter_values = []
	parameter_indexes = []
	transition_str = ['0.0']


	matrix = zeros((nb_states,nb_states),dtype='uint8')
	for t in transitions:
		if type(t[2]) == str:
			val = t[2]
			while '$' in val:
				s = val.index('$')
				e = val.index('$',s+1)
				p = val[s+1:e]
				if not p in parameter_str:
					parameter_str.append(p)
					parameter_values.append(nan)
					parameter_indexes.append([[t[0],t[1]]])
				else:
					parameter_indexes[parameter_str.index(p)].append([t[0],t[1]])
				p_index = parameter_str.index(p)
				val = val[e+1:]
				t[2] = t[2][:s]+'$'+str(p_index)+'$'+t[2][e+1:]
			temp = t[2].replace(' ','')
			if not temp in transition_str:
				transition_str.append(temp)
			matrix[t[0],t[1]] = parameter_str.index(temp)
		elif type(t[2]) == float:
			transition_str.append(str(t[2]))
			matrix[t[0],t[1]] = len(transition_str)-1
		else:
			raise SyntaxError("ERROR")
	for p in parameter_instantiation:
		if not p in parameter_str:
			print("WARNING: no parameter "+p+", instantiation ignored.")
		else:
			parameter_values[parameter_str.index(p)] = parameter_instantiation[p]
	
	return PCTMC(matrix,labeling,transition_str,parameter_values,
				parameter_indexes,parameter_str,name,synchronous_transitions)

def synchronousCompositionPCTMCs(ms: list, name: str = "unknown_composition") -> PCTMC:
	"""
	Returns the synchornous compotision of the PCTMCs in `ms`.
	Parameters
	----------
	ms : list of PCTMCs
		List of PCTMCs to compose.
	name : str, optional.
		Name of the output model.
		Default is 
	Returns
	-------
	PCTMC
		Synchronous composition of `m1` and `m2`.
	"""
	if len(ms) == 0:
		raise ValueError("ms must contain at least 2 models.")
	elif len(ms) == 1:
		return ms[0]
	m1 = ms[0]
	sync_trans = []
	for i in range(1,len(ms)):
		for t in sync_trans:
			m1.synchronous_transitions.append((t[0],t[2],t[1],t[3]))
			m1.matrix[t[0],t[1]] = 0
		print(m1.synchronous_transitions)
		print(m1)
		m1, sync_trans = synchronousComposition2PCTMCs(m1, ms[i], name)
	return m1

def synchronousComposition2PCTMCs(m1: PCTMC, m2: PCTMC, name: str = "unknown_composition"):
	m1_init = [i for i,li in enumerate(m1.labeling) if li == 'init']
	m2_init = [i for i,li in enumerate(m2.labeling) if li == 'init']
	m1_nb_states = m1.nb_states - len(m1_init)
	m2_nb_states = m2.nb_states - len(m2_init)
	nb_states = m1_nb_states * m2_nb_states	
	m1_sids = [i-m1.labeling[:i].count("init") for i,li in enumerate(m1.labeling) if li != 'init']
	m2_sids = [i-m1.labeling[:i].count("init") for i,li in enumerate(m2.labeling) if li != 'init']

	matrix= zeros((nb_states,nb_states),dtype='uint8')
	labeling = []
	p_v = []
	p_str = []
	p_i = []
	trans_str = ['0.0']

	def get_params(s1,s2,model):
		if model == 1:
			m = m1
		else:
			m = m2
		p = []
		ids = m.involvedParameters(s1,s2)
		for i in ids:
			p.append(m.parameter_str[i])
		return p # ['p','q',...]
	
	def get_params_sync_trans(s,model):
		if model == 1:
			m = m1
		else:
			m = m2
		p = []
		while '$' in s:
			start = s.index('$')
			end = s.index('$',start+1)
			p.append(s[start+1:end])
			s = s[end+1:]
		return p

	def get_state_index(s1,s2):
		s2 = m2_sids.index(s2)
		s1 = m1_sids.index(s1)
		return s1*m2_nb_states+s2

	def get_new_trans_string(old_string,model):
		if type(old_string) == float:
			return str(old_string)
		if model == 1:
			m = m1
		else:
			m = m2
		new_string = ''
		while '$' in old_string:
			s = old_string.index("$")
			e = old_string.index('$',s+1)
			new_string += old_string[:s]
			new_string += '$'+str(p_str.index(m.parameter_str[int(old_string[s+1:e])]))+'$'
			old_string = old_string[e+1:]
		new_string += old_string
		return new_string
	
	def get_new_sync_trans_string(old_string):
		if type(old_string) == float:
			return str(old_string)
		new_string = ''
		while '$' in old_string:
			s = old_string.index("$")
			e = old_string.index('$',s+1)
			new_string += old_string[:s]
			new_string += '$'+str(p_str.index(old_string[s+1:e]))+'$'
			old_string = old_string[e+1:]
		new_string += old_string
		return new_string

	def add_in_matrix(trans,s1,s2,model,add_index=[]):
		if model == 1:
			for i in m2_sids:
				x,y = get_state_index(s1,i),get_state_index(s2,i)
				matrix[x,y] = trans
				for ind in add_index:
					p_i[ind].append([x,y])
		elif model == 2:
			for i in m1_sids:
				x,y = get_state_index(i,s1),get_state_index(i,s2)
				matrix[x,y] = trans
				for ind in add_index:
					p_i[ind].append([x,y])

	def add_float_transition(s1,s2,model):
		trans_str.append('$'+str(len(p_str))+'$')
		if model == 1:
			p_str.append("p_unamed_"+str(s1)+'_'+str(s2))
			p_v.append(m1.transitionValue(s1,s2))
		else:
			p_str.append("q_unamed_"+str(s1)+'_'+str(s2))
			p_v.append(m2.transitionValue(s1,s2))
		p_i.append([])
		add_in_matrix(trans=len(trans_str)-1,s1=s1,s2=s2,model=model,add_index=[len(p_i)-1])

	def add_named_parameter(s1,s2,model):
		if model == 1:
			m = m1
		else:
			m = m2
		p = get_params(s1,s2,model)
		add_index=[]
		for k in p:
			if not k in p_str:
				p_str.append(k)
				p_v.append(m.parameterValue(k))
				p_i.append([])
			add_index.append(p_str.index(k))
		old_string = m.transition_str(s1,s2)
		new_string = get_new_trans_string(old_string,model)
		if not new_string in trans_str:
			trans_str.append(new_string)
		add_in_matrix(trans_str.index(new_string),s1,s2,model,add_index)

	for s1 in m1_sids: # labeling
		for s2 in m2_sids:
			labeling.append(m1.labeling[s1]+','+m2.labeling[s2])

	for i in m1_sids: # m1 transitions
		for j in m1_sids:
			if m1.matrix[i,j] == 0:
				add_in_matrix(trans=0,s1=i,s2=j,model=1)
			elif '$' in m1.transitionStr(i,j):
				add_named_parameter(i,j,1)
			else:
				add_float_transition(i,j,1)

	for i in m2_sids: # m2 transitions
		for j in m2_sids:
			if m2.matrix[i,j] == 0.0:
				add_in_matrix(trans=0,s1=i,s2=j,model=2)
			elif '$' in m2.transitionStr(i,j):
				add_named_parameter(i,j,2)
			else:
				add_float_transition(i,j,2)

	for i in m1_sids: # self loops
		if m1.matrix[i,i] != 0:
			for j in m2_sids:
				if m2.matrix[j,j] != 0:
					if not '$' in m1.transitionStr(i,i):
						t_str = '$'+str(p_str.index("p_unamed_"+str(i)+'_'+str(i)))+'$*'
					else:
						t_str = '('+get_new_trans_string(m1.transition_str(i,i),1)+')*'
					if not '$' in m2.transitionStr(j,j):
						t_str = '$'+str(p_str.index("q_unamed_"+str(j)+'_'+str(j)))+'$*'
					else:
						t_str += '('+get_new_trans_string(m2.transition_str(j,j),2)+')'
					if not t_str in trans_str:
						trans_str.append(t_str)

	sync_trans = []
	for sync_1 in m1.synchronous_transitions: # synchronous transitions
		si,ai,di,pi = sync_1
		for sync_2 in m2.synchronous_transitions:
			sj,aj,dj,pj = sync_2
			if ai == aj:
				if type(pi) != float or type(pj) != float:
					ps = []
					if type(pi) != float:
						ps += get_params_sync_trans(pi,1)
					for i in ps:
						if not i in p_str:
							p_str.append(i)
							p_v.append(m1.parameterValue(i))
							p_i.append([])
					if type(pj) != float:
						ps += get_params_sync_trans(pj,2)
					for i in ps:
						if not i in p_str:
							p_str.append(i)
							p_v.append(m2.parameterValue(i))
							p_i.append([])
					for i in ps:
						indexes = [get_state_index(si,sj),get_state_index(di,dj)]
						if not indexes in p_i[p_str.index(i)]:
							p_i[p_str.index(i)].append(indexes)
					prev_val = matrix[get_state_index(si,sj),get_state_index(di,dj)]
					if prev_val == 0:
						matrix[get_state_index(si,sj),get_state_index(di,dj)] = len(trans_str)
						trans_str.append('('+get_new_sync_trans_string(pi)+')*('+get_new_sync_trans_string(pj)+')')
						prev_val = -1
					else:
						trans_str[prev_val]+= '+ ('+get_new_sync_trans_string(pi)+')*('+get_new_sync_trans_string(pj)+')'
					sync_trans.append((get_state_index(si,sj),get_state_index(di,dj),ai,'('+str(pi)+')*('+str(pj)+')'))
				else:
					matrix[get_state_index(si,sj),get_state_index(di,dj)] = len(trans_str)
					trans_str.append(str(pi*pj))
					sync_trans.append((get_state_index(si,sj),get_state_index(di,dj),ai,pi*pj))
				
	labeling.append('init')
	matrix = vstack((matrix,zeros(nb_states,dtype='uint8')))
	matrix = hstack((matrix,zeros(nb_states+1,dtype='uint8')[:,newaxis]))
	m1_init_trans = zeros(m1_nb_states)
	for i in m1_init:
		tmp = [m1.transitionValue(i,j) for j in m1_sids]
		for j in range(m1_nb_states):
			if m1_init_trans[j]*tmp[j]>0.0:
				m1_init_trans[j] *= tmp[j]
			else:
				m1_init_trans[j] += tmp[j]
	m2_init_trans = zeros(m2_nb_states)
	for i in m2_init:
		tmp = [m2.transitionValue(i,j) for j in m2_sids]
		for j in range(m2_nb_states):
			if m2_init_trans[j]*tmp[j]>0.0:
				m2_init_trans[j] *= tmp[j]
			else:
				m2_init_trans[j] += tmp[j]
	for i,si in enumerate(m1_init_trans):
		for j,sj in enumerate(m2_init_trans):
			if si*sj == 0.0:
				matrix[-1][get_state_index(i,j)] = 0
			else:
				matrix[-1][get_state_index(i,j)] = len(trans_str)
				trans_str.append(str(si*sj))

	i = 0
	while i < len(matrix): # removing unreachable state
		if (matrix.T[i] == 0).all() == True and labeling[i] != 'init':
			nb_states -= 1
			matrix = delete(matrix,i,0)
			matrix = delete(matrix,i,1)
			labeling = labeling[:i]+labeling[i+1:]
			for j in range(len(p_i)):
				k = 0
				while k < len(p_i[j]): 
					if p_i[j][k][0] == i:
						p_i[j].remove(p_i[j][k])
					else:
						if p_i[j][k][0] > i:
							p_i[j][k][0] -= 1
						if p_i[j][k][1] > i:
							p_i[j][k][1] -= 1
						k += 1
			j = 0
			while j <len(sync_trans):
				if sync_trans[j][0] == i or sync_trans[j][1] == i:
					sync_trans = sync_trans[:j]+sync_trans[j+1:]
					j -= 1
				j += 1
			i = -1
		i += 1
	p_i[0] = []

	to_remove = list(set(range(len(trans_str))) - set(matrix.flatten()) )
	for i in to_remove:
		trans_str = trans_str[:i]+trans_str[i+1:]
		for j in range(len(matrix)):
			for k in range(len(matrix)):
				if matrix[j][k] > i:
					matrix[j][k] -= 1
		for j in range(len(to_remove)):
			if to_remove[j] > i:
				to_remove[j] -= 1

	return PCTMC(matrix,labeling,trans_str,p_v,p_i,p_str,name), sync_trans
