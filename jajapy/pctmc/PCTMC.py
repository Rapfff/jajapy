from ..base.tools import resolveRandom
from ..base.Parametric_Model import *
from numpy import  zeros, dot, newaxis, hstack, inf, vstack, delete
from numpy.random import exponential, rand, seed
from math import exp, log
from multiprocessing import cpu_count, Pool
from sys import platform

class PCTMC(Parametric_Model):
	"""
	Class representing a PCTMC.
	"""
	def __init__(self, matrix: ndarray, labelling: list,
				 transition_expr: list, parameter_values: dict,
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
		labelling: list of str
			A list of N observations (with N the nb of states).
			If `labelling[s] == o` then state of ID `s` is labelled by `o`.
			Each state has exactly one label.
		transition_str: list of str
			Contains the symbolic value for each transition.
		parameter_values: dict
			Contains the value for each instantiated parameter.
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
		self.model_type = PCTMC_ID
		self.synchronous_transitions = synchronous_transitions
		super().__init__(matrix,labelling,transition_expr,parameter_values,parameter_indexes,parameter_str,name)

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
		raise ValueError("cannot compute the exit rate of non-instantiated state "+str(s))

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
		if self.labelling[s1] != obs:
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
		try:
			return self.e(s)*exp(-self.e(s)*t)
		except ValueError:
			print("WARN: cannot compute the lkl of non-instantiated state "+str(s))

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
		try:
			e = self.e(s1)
			if e == 0.0:
				return inf
			return self.l(s1,s2, obs)/e
		except ValueError:
			print("WARN: cannot compute the exit-rate of non-instantiated state "+str(s1))
	
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
		try:
			e = self.e(s)
			if e == 0.0:
				return inf
			return 1/e
		except ValueError:
			print("WARN: cannot compute the expected time of non-instantiated state "+str(s))

	def _checkRates(self,values=None) -> bool:
		if type(values) == type(None):
			values = self.parameter_values
		xs,ys = where(self.matrix != 0)
		for x,y in zip(xs,ys):
			v = self.evaluateTransition(x,y)
			if type(v) == float:
				if v < 0.0:
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
		
		Returns
		-------
		bool
			True if the instantiation is valid (no transition has a negative
			evaluation) and then applied, False if the instantiation is not 
			valid (and then ignored).
		"""
		new_values =  super().instantiate(parameters, values)
		if self._checkRates(new_values):
			self.parameter_values = new_values
			return True
		return False

	def randomInstantiation(self, parameters: list = None,min_val:float = None,
			 				max_val:float = None, sseed: int = None) -> None:
		"""
		Randomly instantiated the parameters given in `parameters`.
		If `parameters` is not set it instantiates all the non-instantiated
		parameters in the model.

		Parameters
		----------
		parameters : list of string, optional.
			List of parameters names.
		min_val : float, optional
			Minimal value for the randomly instantiated parameters.
			If not set and if the model has at least two instantiated parameters,
			this value is equal to the parameters with the smallest instantiation.
			If not set and if the model has less than two instantiated parameters,
			this value is equal to 0.1.
		max_val : float, optional
			Maximal value for the randomly instantiated parameters.
			If not set and if the model has at least two instantiated parameters,
			this value is equal to the parameters with the highest instantiation.
			If not set and if the model has less than two instantiated parameters,
			this value is equal to 5.0.
		sseed : int, optional
			the seed value.
		"""
		if parameters == None:
			parameters = []
			for i in self.parameter_str:
				if not i in self.parameter_values:
					parameters.append(i)
				elif isnan(self.parameter_values[i]):
					parameters.append(i)
		if min_val == None:
			if len(list(self.parameter_values.values())) > 1:
				min_val = min(self.parameter_values.values())
			else:
				min_val = 0.01
		if max_val == None:
			if len(list(self.parameter_values.values())) > 1:
				max_val = max(self.parameter_values.values())
			else:
				max_val = 0.3
	
		if sseed != None:
			seed(sseed)
		val = rand(len(parameters))*(max_val-min_val)+min_val
		while not self.instantiate(parameters,val):
			val = rand(len(parameters))*(max_val-min_val)+min_val

		seed()

	def _stateToString(self,state:int) -> str:
		res = "----STATE "+str(state)+"--"+self.labelling[state]+"----\n"
		if self.isInstantiated(state):
			res += "Exepected waiting time: "+str(round(self.expected_time(state),5))+'\n'
		for j in range(len(self.matrix[state])):
			if self.matrix[state][j] != 0:
				val = str(self.transitionExpression(state,j))
				if self.isInstantiated(state,j) and len(self.involvedParameters(state,j)) > 0:
					val += ' (='+str(round(self.transitionValue(state,j),5))+')'
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
		next_obs  = self.labelling[state]
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
		output.append(self.labelling[current])
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
		last_arr = prev_arr * (array(self.labelling) == obs_seq[-1])
		return log(last_arr.sum())*times

	def logLikelihood(self,traces: Set) -> float:
		"""
		Computes the loglikelihood of `traces` under this model.

		Parameters
		----------
		traces : Set
			a set of traces.

		Returns
		-------
		float
			the loglikelihood of `traces` under this model.
		"""
		if not self.isInstantiated():
			raise ValueError("Cannot compute the loglikelihood of a set under a non-instantiated model.")
			
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
	
	def savePrism(self,file_path:str) -> None:
		"""
		Save this model into `file_path` in the Prism format.

		Parameters
		----------
		file_path : str
			Path of the output file.
		"""
		f = open(file_path,'w')
		f.write("ctmc\n\n")

		for p in self.parameter_str:
			f.write("const double "+p)
			if self.isInstantiated(param=p):
				f.write(' = '+str(self.parameterValue(p)))
			f.write(';\n')

		f.write("\nmodule "+self.name+'\n')
		f.write("\ts: [0.."+str(self.nb_states)+"] init "+str(where(self.initial_state==1.0)[0][0])+";\n\n")
		
		for s1 in range(self.nb_states):
			if (self.matrix[s1]!=0).any():
				res = '\t[] s='+str(s1)+' ->'
				for s2 in where(self.matrix[s1] != 0)[0]:
					t = str(self.transition_expr[self.matrix[s1,s2]])
					t = t.replace('**','^')
					res += ' '+t+" : (s'="+str(s2)+") +"
				res = res[:-2]+';\n'
				f.write(res)
		f.write('endmodule\n\n')

		labels = {}
		for s,l in enumerate(self.labelling):
			if l != 'init':
				if not l in labels:
					labels[l] = [str(s)]
				else:
					labels[l].append(str(s))
		for l in labels:
			res = 'label "'+l+'" ='
			for s in labels[l]:
				res += ' s='+s+' |'
			res = res[:-1]+';\n'
			f.write(res)
		f.close()
		
	
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
	if l != "PCTMC":
		msg = "This file doesn't describe a PCTMC: it describes a "+l
		raise ValueError(msg)
	matrix,labelling,parameter_values,parameter_indexes,parameter_str,transition_expr,name = loadParametricModel(f)
	f.close()
	return PCTMC(matrix,labelling,transition_expr,parameter_values,
				 parameter_indexes,parameter_str,name)

def createPCTMC(transitions: list, labelling: list, parameters: list,
				initial_state, parameter_instantiation: dict={},
				synchronous_transitions=[], name: str ="unknown_PCTMC") -> PCTMC:
	"""
	An user-friendly way to create a PCTMC.

	Parameters
	----------
	transitions : [ list of tuples (int, int, float or str)]
		Each tuple represents a transition as follow: 
		(source state ID, destination state ID, probability).
		The probability can be explicitly given (then it's a float),
		or a parameter (then it's the name of the parameter).
	labelling: list of str
		A list of N observations (with N the nb of states).
		If `labelling[s] == o` then state of ID `s` is labelled by `o`.
		Each state has exactly one label.
	parameters: list of str.
		A list containing all the parameters name.
	initial_state : int or list of float
		Determine which state is the initial one (then it's the id of the
		state), or what are the probability to start in each state (then it's
		a list of probabilities).
	parameter_instantiation: dict
		An instantion for some (or all) parameters.
		`parameter_instantiation == {'p':0.5}` means that parameter `p`
		should be instantiated to 0.5. The other parameters are not
		instantiated.
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
		the PCTMC describes by `transitions`, `labelling`, and `initial_state`.
	"""
	if 'init' in labelling:
		msg =  "The label 'init' cannot be used: it is reserved for initial states."
		raise SyntaxError(msg)
	
	labelling.append('init')
	
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
	if nb_states > len(labelling):
		raise ValueError("All states are not labelled (the labelling list is too small).")
	elif nb_states < len(labelling):
		print("WARNING: the labelling list is bigger than the number of states")
	
	parameter_str = parameters
	parameter_values = {}
	parameter_indexes = [[] for _ in parameter_str]
	transition_expr = [sympify('0.0')]

	matrix = zeros((nb_states,nb_states),dtype='uint16')
	for t in transitions:
		val = str(t[2])
		expr = sympify(val)
		if expr.is_real or not expr in transition_expr:
			transition_expr.append(expr)
		matrix[t[0],t[1]] = transition_expr.index(expr)
		for p in expr.free_symbols:
			parameter_indexes[parameter_str.index(str(p))].append([t[0],t[1]])

	for p in parameter_instantiation:
		if not p in parameter_str:
			print("WARNING: no parameter "+p+", instantiation ignored.")
		else:
			parameter_values[p] = parameter_instantiation[p]
	return PCTMC(matrix,labelling,transition_expr,parameter_values,
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
		Default is unknown_composition.

	Returns
	-------
	PCTMC
		Synchronous composition of `m1` and `m2`.
	"""
	def removeUnreachableState(m):
		i = 0
		while i < len(m.matrix): # removing unreachable state
			if (m.matrix.T[i] == 0).all() == True and m.labelling[i] != 'init':
				m.nb_states -= 1
				m.matrix = delete(m.matrix,i,0)
				m.matrix = delete(m.matrix,i,1)
				m.initial_state = delete(m.initial_state,i)
				m.labelling = m.labelling[:i]+m.labelling[i+1:]
				for j in range(len(m.parameter_indexes)):
					k = 0
					while k < len(m.parameter_indexes[j]): 
						if i in m.parameter_indexes[j][k]:
							m.parameter_indexes[j] = m.parameter_indexes[j][:k]+m.parameter_indexes[j][k+1:]
						else:
							if m.parameter_indexes[j][k][0] > i:
								m.parameter_indexes[j][k][0] -= 1
							if m.parameter_indexes[j][k][1] > i:
								m.parameter_indexes[j][k][1] -= 1
							k += 1
				i = -1
			i += 1

	def removeUnusedTrans(m):
		to_remove = list(set(range(len(m.transition_expr))) - set(m.matrix.flatten()) )
		for i in to_remove:
			m.transition_expr = m.transition_expr[:i]+m.transition_expr[i+1:]
			for j in range(len(m.matrix)):
				for k in range(len(m.matrix)):
					if m.matrix[j][k] > i:
						m.matrix[j][k] -= 1
			for j in range(len(to_remove)):
				if to_remove[j] > i:
					to_remove[j] -= 1

	if len(ms) == 0:
		raise ValueError("ms must contain at least 2 models.")
	elif len(ms) == 1:
		return ms[0]
	m1 = ms[0]
	sync_trans = []
	for i in range(1,len(ms)):
		for t in sync_trans:
			m1.synchronous_transitions.append(t)
		m1, sync_trans = synchronousComposition2PCTMCs(m1, ms[i], name)
	# add sync trans
	for t in sync_trans:
		i, _, d, expr = t
		if m1.matrix[i,d] != 0.0:
			expr = expr + m1.transitionExpression(i,d)
		if not expr in m1.transition_expr:
			m1.transition_expr.append(expr)
		m1.matrix[i,d] = m1.transition_expr.index(expr)
		for p in expr.free_symbols:
			m1.parameter_indexes[m1.parameter_str.index(p)].append([i,d])
	removeUnreachableState(m1)
	removeUnusedTrans(m1)
	return m1

def synchronousComposition2PCTMCs(m1: PCTMC, m2: PCTMC, name: str = "unknown_composition") -> PCTMC:
	"""
	Returns the synchronous composition of `m1` and `m2`.

	Parameters
	----------
	m1 : CTMC
		First CTMC to compose with.
	m2 : CTMC
		Second CTMC to compose with.

	Returns
	-------
	PCTMC
		Synchronous composition of `m1` and `m2`.
	"""
	m1_init = [i for i,li in enumerate(m1.labelling) if li == 'init']
	m2_init = [i for i,li in enumerate(m2.labelling) if li == 'init']
	m1_nb_states = m1.nb_states - len(m1_init)
	m2_nb_states = m2.nb_states - len(m2_init)
	nb_states = m1_nb_states * m2_nb_states	
	m1_sids = [i-m1.labelling[:i].count("init") for i,li in enumerate(m1.labelling) if li != 'init']
	m2_sids = [i-m1.labelling[:i].count("init") for i,li in enumerate(m2.labelling) if li != 'init']

	matrix= zeros((nb_states,nb_states),dtype='uint16')
	labelling = []
	p_v = m1.parameter_values #TODO m2.parameter_values
	p_str = list(set(m1.parameter_str+m2.parameter_str))
	p_i = [[] for _ in p_str]
	trans_expr = [sympify('0.0')]

	def get_state_index(s1,s2):
		s2 = m2_sids.index(s2)
		s1 = m1_sids.index(s1)
		return s1*m2_nb_states+s2

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
	
	def add_transition(i,j,model):
		if model == 1:
			m = m1
		else:
			m = m2
		expr = m.transitionExpression(i,j)
		if not expr in trans_expr:
			trans_expr.append(expr)
		add_index = []
		for p in m.involvedParameters(i,j):
			if not p in p_str:
				p_str.append(p)
			add_index.append(p_str.index(p))
		add_in_matrix(trans_expr.index(expr),i,j,model,add_index)
	
	def add_sync_transition(a,s1, d1, expr1, s2, d2, expr2):
		expr1 = sympify(expr1)
		expr2 = sympify(expr2)
		expr = expr1*expr2
		for p in expr.free_symbols:
			if not p in p_str:
				p_str.append(p)
				p_i.append([])
		x, y = get_state_index(s1,s2), get_state_index(d1, d2)
		return [x,a,y,expr]

	for s1 in m1_sids: # labelling
		for s2 in m2_sids:
			labelling.append(m1.labelling[s1]+','+m2.labelling[s2])
	for i in m1_sids: # m1 transitions
		for j in m1_sids:
			add_transition(i,j,1)
	for i in m2_sids: # m2 transitions
		for j in m2_sids:
			add_transition(i,j,2)
	for i in m1_sids: # self loops
		if m1.matrix[i,i] != 0:
			for j in m2_sids:
				if m2.matrix[j,j] != 0:
					expr = m1.transitionExpression(i,i)+m2.transitionExpression(j,j)
					if not expr in trans_expr:
						trans_expr.append(expr)
					matrix[get_state_index(i,j),get_state_index(i,j)] = trans_expr.index(expr)
	sync_trans = []
	for sync_1 in m1.synchronous_transitions: # synchronous transitions
		si,ai,di,pi = sync_1
		for sync_2 in m2.synchronous_transitions:
			sj,aj,dj,pj = sync_2
			if ai == aj:
				sync_trans.append(add_sync_transition(ai,si,di,pi,sj,dj,pj))
				
	labelling.append('init')
	matrix = vstack((matrix,zeros(nb_states,dtype='uint16')))
	matrix = hstack((matrix,zeros(nb_states+1,dtype='uint16')[:,newaxis]))
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
				matrix[-1][get_state_index(i,j)] = len(trans_expr)
				trans_expr.append(sympify(str(si*sj)))

	return PCTMC(matrix,labelling,trans_expr,p_v,p_i,p_str,name), sync_trans
