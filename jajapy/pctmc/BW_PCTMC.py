from .PCTMC import *
from ..base.BW import BW
from ..base.Set import Set
from numpy import array, zeros, dot, append, ones, log, inf, uint16, max
from numpy.polynomial.polynomial import polyroots


class BW_PCTMC(BW):
	def __init__(self) -> None:
		super().__init__()
	
	def fit(self, traces, initial_model: PCTMC, output_file: str=None,
			epsilon: float=0.01, max_it: int= inf, pp: str='',
			verbose: bool = True, return_data: bool = False,
			stormpy_output: bool = True, fixed_parameters: list = [],
			update_constant :bool = True) -> PCTMC:
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : Set or list or numpy.ndarray
			training set.
		initial_model : PCTMC
			first hypothesis.
		output_file : str, optional
			if set path file of the output model. Otherwise the output model
			will not be saved into a text file.
		epsilon : float, optional
			the learning process stops when the difference between the
			loglikelihood of the training set under the two last hypothesis is
			lower than ``epsilon``. The lower this value the better the output,
			but the longer the running time. By default 0.01.
		max_it: int
			Maximal number of iterations. The algorithm will stop after `max_it`
			iterations.
			Default is infinity.
		pp : str, optional
			Will be printed at each iteration. By default ''.
		verbose: bool, optional
			Print or not a small recap at the end of the learning.
			Default is True.
		return_data: bool, optional
			If set to True, a dictionary containing following values will be
			returned alongside the hypothesis once the learning is done.
			'learning_rounds', 'learning_time', 'training_set_loglikelihood'.
			Default is False.
		stormpy_output: bool, optional
			If set to True the output model will be a Stormpy sparse model.
			Default is True.
		fixed_parameters: list of str, optional
			list of parameter names. These parameters will not be updated.
			By default no parameters will be fixed (i.e. the list is empty).
		update_constant: bool, optional
			If set to False, the constant transitions (i.e. tha transition
			that doesn't depend on any parameter) will no be updated.
			Default is True.

		Returns
		-------
		PCTMC or stormpy.SparseParametricCtmc
			The fitted PCTMC.
			If `stormpy_output` is set to `False` or if stormpy is not available on
			the machine it returns a `jajapy.PCTMC`, otherwise it returns a `stormpy.SparseParametricCtmc`
		"""
		if type(traces) != Set:
			traces = Set(traces)
		
		alphabet = traces.getAlphabet()
		if not 'init' in alphabet:
			alphabet.append('init')
			timed = type(traces.sequences[0][1]) != str
			for s in range(len(traces.sequences)):
				if timed:
					traces.sequences[s].insert(0,0.5)
				traces.sequences[s].insert(0,'init')
		self.h = initial_model

		try:
			from ..with_stormpy import stormpyModeltoJajapy
			stormpy_installed = True
		except ModuleNotFoundError:
			stormpy_installed = False

		try:
			initial_model.name
		except AttributeError: # then initial_model is a stormpy sparse model
			if not stormpy_installed:
				raise RuntimeError("the initial model is a Storm model but Storm is not installed on the machine")
			initial_model = stormpyModeltoJajapy(initial_model)	

		initial_model.randomInstantiation()
		self.nb_parameters = initial_model.nb_parameters

		self.update_constant = update_constant

		self.sortParameters(fixed_parameters)

		return super().fit(traces, initial_model, output_file, epsilon, max_it, pp, verbose,return_data,stormpy_output)
	
	def fit_nonInstantiatedParameters(self, traces, initial_model: PCTMC,
			output_file: str=None, epsilon: float=0.01, max_it: int= inf,
			pp: str='', verbose: bool = True, return_data: bool = False,
			stormpy_output: bool = True) -> PCTMC:
		"""
		Fits only the non-instantiated parameters in the initial model
		according to ``traces``.

		Parameters
		----------
		traces : Set or list or numpy.ndarray
			training set.
		initial_model : PCTMC
			first hypothesis.
		output_file : str, optional
			if set path file of the output model. Otherwise the output model
			will not be saved into a text file.
		epsilon : float, optional
			the learning process stops when the difference between the
			loglikelihood of the training set under the two last hypothesis is
			lower than ``epsilon``. The lower this value the better the output,
			but the longer the running time. By default 0.01.
		max_it: int
			Maximal number of iterations. The algorithm will stop after `max_it`
			iterations.
			Default is infinity.
		pp : str, optional
			Will be printed at each iteration. By default ''.
		verbose: bool, optional
			Print or not a small recap at the end of the learning.
			Default is True.
		return_data: bool, optional
			If set to True, a dictionary containing following values will be
			returned alongside the hypothesis once the learning is done.
			'learning_rounds', 'learning_time', 'training_set_loglikelihood'.
			Default is False.
		stormpy_output: bool, optional
			If set to True the output model will be a Stormpy sparse model.
			Default is True.

		Returns
		-------
		PCTMC or stormpy.SparseParametricCtmc
			The fitted PCTMC.
			If `stormpy_output` is set to `False` or if stormpy is not available on
			the machine it returns a `jajapy.PCTMC`, otherwise it returns a `stormpy.SparseParametricCtmc`
		"""
		update_constant = False
		fixed_parameters = []
		print("Fitting only the non-instantiated parameters, i.e: ",end='')
		
		to_update = []
		for p in initial_model.parameter_str:
			if initial_model.isInstantiated(param=p):
				fixed_parameters.append(p)
			else:
				to_update.append(p)
		print(', '.join(to_update))

		# remove all the states never used in the training set:
		# we only care here about the values of some parameters,
		# the structure of the model is out of scope.
		if type(traces) != Set:
			traces = Set(traces)
		alphabet = traces.getAlphabet()
		print("Removing unused states...")
		print('From',initial_model.nb_states,'states to',end=' ')
		s = 0
		while s < initial_model.nb_states:
			if initial_model.getLabel(s) not in alphabet:
				initial_model.nb_states -= 1
				initial_model.labeling = initial_model.labeling[:s]+initial_model.labeling[s+1:]
				initial_model.initial_state = delete(initial_model.initial_state,s)
				initial_model.matrix = delete(initial_model.matrix,s,0)
				initial_model.matrix = delete(initial_model.matrix,s,1)
				for j in range(len(initial_model.parameter_indexes)):
					k = 0
					while k < len(initial_model.parameter_indexes[j]): 
						if s in initial_model.parameter_indexes[j][k]:
							initial_model.parameter_indexes[j] = initial_model.parameter_indexes[j][:k]+initial_model.parameter_indexes[j][k+1:]
						else:
							if initial_model.parameter_indexes[j][k][0] > s:
								initial_model.parameter_indexes[j][k][0] -= 1
							if initial_model.parameter_indexes[j][k][1] > s:
								initial_model.parameter_indexes[j][k][1] -= 1
							k += 1
				s = -1
			s += 1
		t = 1
		l = len(initial_model.transition_expr)
		while t < l:
			if not (initial_model.matrix == t).any():
				initial_model.transition_expr = initial_model.transition_expr[:t]+initial_model.transition_expr[t+1:]
				initial_model.matrix -= (initial_model.matrix > t)
				l -= 1
			else:
				t += 1
		initial_model.alphabet = alphabet
		print(initial_model.nb_states)
		self.fit(traces,initial_model,output_file,epsilon,max_it,pp,
				verbose, return_data,stormpy_output,fixed_parameters,update_constant)
		
		res = {}
		for p in to_update:
			res[p] = self.h.parameterValue(p)
		return res
	
	def fit_component(self, traces, m1: PCTMC, m2: PCTMC,
			output_file: str=None, epsilon: float=0.01, max_it: int= inf,
			pp: str='', verbose: bool = True, return_data: bool = False,
			stormpy_output: bool = True) -> PCTMC:
		"""
		Given a instantiated model `m1` and a model `m2`, updates `m2` to
		maximise the loglikelihood of `traces` under the composition of
		`m1` and `m2`. 
		
		Parameters
		----------
		traces : Set or list or numpy.ndarray
			training set.
		initial_model : PCTMC
			first hypothesis.
		output_file : str, optional
			if set path file of the output model. Otherwise the output model
			will not be saved into a text file.
		epsilon : float, optional
			the learning process stops when the difference between the
			loglikelihood of the training set under the two last hypothesis is
			lower than ``epsilon``. The lower this value the better the output,
			but the longer the running time. By default 0.01.
		max_it: int
			Maximal number of iterations. The algorithm will stop after `max_it`
			iterations.
			Default is infinity.
		pp : str, optional
			Will be printed at each iteration. By default ''.
		verbose: bool, optional
			Print or not a small recap at the end of the learning.
			Default is True.
		return_data: bool, optional
			If set to True, a dictionary containing following values will be
			returned alongside the hypothesis once the learning is done.
			'learning_rounds', 'learning_time', 'training_set_loglikelihood'.
			Default is False.
		stormpy_output: bool, optional
			If set to True the output model will be a Stormpy sparse model.
			Default is True.

		Returns
		-------
		PCTMC or stormpy.SparseParametricCtmc
			The fitted PCTMC.
			If `stormpy_output` is set to `False` or if stormpy is not available on
			the machine it returns a `jajapy.PCTMC`, otherwise it returns a `stormpy.SparseParametricCtmc`
		"""
		pass

	def sortParameters(self,fixed_parameters):
		self.a_pis = zeros((self.h.nb_states,self.h.nb_states,len(self.h.parameter_str)),dtype=uint16)
		for iparam,param in enumerate(self.h.parameter_str):
			if not param in fixed_parameters:
				for s,ss in self.h.parameterIndexes(param):
					self.a_pis[s,ss,iparam] = self.a_pi(s,ss,param)

		self.parameters_cat = [[],[],[]]
		for ip,p in enumerate(self.h.parameter_str):
			apis = [self.a_pis[x,y,ip] for x,y in self.h.parameterIndexes(p)]
			cs = [self.C(x,y) for x,y in self.h.parameterIndexes(p)]
			if min(apis) == 1 and max(apis) == 1 and min(cs) == 1 and max(cs) == 1:
				self.parameters_cat[0].append(p)
			elif min(cs) == max(cs):
				self.parameters_cat[1].append(p)
			else:
				self.parameters_cat[2].append(p)
		
		self.c_pis = zeros((self.h.nb_states,self.h.nb_states,len(self.parameters_cat[0])),dtype=uint16)
		for iparam,param in enumerate(self.parameters_cat[0]):
			for s,ss in self.h.parameterIndexes(param):
				self.c_pis[s,ss,iparam] = self.c_pi(s,ss,param)
	
	def _computeTaus(self):
		self.hval = zeros((self.nb_states,self.nb_states))
		for s in range(self.nb_states):
			for ss in range(self.nb_states):
				self.hval[s,ss] =  self.h.evaluateTransition(s,ss)

	def a_pi(self,s1,s2,p):
		t = self.h.transitionExpression(s1,s2)
		while not t.is_Pow and not t.is_Symbol:
			flag = False
			for a in t.args:
				if p in [x.name for x in a.free_symbols]:
					t = a
					flag = True
					break
			if not flag:
				print(p,'not in transition',s1,s2)
				input()
		if t.is_Pow:
			return t.args[1]
		else:
			return 1

	def c_pi(self,s1,s2,p):
		t = self.h.transitionExpression(s1,s2)
		while not t.is_Mul and not t.is_Symbol:
			flag = False
			for a in t.args:
				if p in [x.name for x in a.free_symbols]:
					t = a
					flag = True
					break
			if not flag:
				print(p,'not in transition',s1,s2)
				input()
		if t.is_Mul:
			return t.args[0]
		else:
			return 1

	def C(self,s1,s2):
		r = 0
		for p in self.h.involvedParameters(s1,s2):
			r += self.a_pis[s1,s2,self.h.parameter_str.index(p)]
		return r
		
	def h_e(self,s: int) -> float:
		"""
		Returns the exit rate, in the current hypothesis, of state ``s``, i.e.
		the sum of all the rates in this state.

		Returns
		-------
		s : int
			A state ID.
		float
			An exit rate.
		"""
		return self.hval[s].sum()
	
	def h_l(self, s1: int, s2: int, obs: str) -> float:
		"""
		Returns the rate, in the current hypothesis, associated to the
		transition from state `s1` to state `s2` generating observation `obs`.

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
		if self.h.labeling[s1] != obs:
			return 0.0
		return self.hval[s1,s2]

	def h_tau(self,s1: int,s2: int,obs: str) -> float:
		"""
		Probability of moving from ``s1`` to ``s2`` generating ``obs`` under
		the current hypothesis.

		Parameters
		----------
		s1 : int
			source state ID.
		s2 : int
			destination state ID.
		obs : str
			observation.

		Returns
		-------
		float:
			probability of moving from ``s1`` to ``s2`` generating ``obs``.
		"""
		if self.h.labeling[s1] != obs:
			return 0.0
		e = self.h_e(s1)
		if e == 0.0:
			return inf
		return self.hval[s1,s2]/e


	def computeAlphas(self,obs_seq: list, times_seq: list = None) -> array:
		if times_seq:
			return self.computeAlphas_timed(obs_seq,times_seq)
		else:
			return self.computeAlphas_nontimed(obs_seq)

	def computeBetas(self,obs_seq: list, times_seq: list = None) -> array:
		if times_seq:
			return self.computeBetas_timed(obs_seq,times_seq)
		else:
			return self.computeBetas_nontimed(obs_seq)

	def computeAlphas_timed(self,obs_seq: list, times_seq: list) -> array:
		"""
		Compute the beta values for ``obs_seq`` and ``times_seq`` under the
		current BW hypothesis.

		Parameters
		----------
		obs_seq : list of str
			Sequence of observations.
		times_seq : list of float
			Sequence of waiting times.

		Returns
		-------
		2-D narray
			array containing the beta values.
		"""
		len_seq = len(obs_seq)-1
		init_arr = self.h.initial_state
		zero_arr = zeros(shape=(len_seq*self.nb_states,))
		alpha_matrix = append(init_arr,zero_arr)
		alpha_matrix = alpha_matrix.reshape(len_seq+1,self.nb_states)
		for k in range(len_seq):
			for s in range(self.nb_states):
				p = array([self.h_l(ss,s,obs_seq[k])*exp(-self.h_e(ss)*times_seq[k]) for ss in range(self.nb_states)])
				alpha_matrix[k+1,s] = dot(alpha_matrix[k],p)
		alpha_matrix[-1] *= (array(self.h.labeling) == obs_seq[-1])
		print(alpha_matrix)
		return alpha_matrix.T

	def computeBetas_timed(self,obs_seq: list, times_seq: list) -> array:
		"""
		Compute the beta values for ``obs_seq`` and ``times_seq`` under the
		current BW hypothesis.

		Parameters
		----------
		obs_seq : list of str
			Sequence of observations.
		times_seq : list of float
			Sequence of waiting times.

		Returns
		-------
		2-D narray
			array containing the beta values.
		"""
		len_seq = len(obs_seq)-1
		init_arr = ones(self.nb_states)*(array(self.h.labeling) == obs_seq[-1])
		zero_arr = zeros(shape=(len_seq*self.nb_states,))
		beta_matrix = append(zero_arr,init_arr).reshape(len_seq+1,self.nb_states)
		for k in range(len_seq-1,-1,-1):
			for s in range(self.nb_states):
				p = array([self.h_l(s,ss,obs_seq[k]) for ss in range(self.nb_states)])
				p = p * exp(-self.h_e(s)*times_seq[k])
				beta_matrix[k,s] = dot(beta_matrix[k+1],p)
		return beta_matrix.T

	def computeAlphas_nontimed(self,sequence: list) -> array:
		"""
		Compute the beta values for ``sequence`` under the current BW
		hypothesis.

		Parameters
		----------
		sequence : list of str
			sequence of observations.

		Returns
		-------
		2-D narray
			array containing the beta values.
		"""
		return super().computeAlphas(sequence)

	def computeBetas_nontimed(self,sequence: list) -> array:
		"""
		Compute the beta values for ``sequence`` under the current BW
		hypothesis.

		Parameters
		----------
		sequence : list of str
			sequence of observations.

		Returns
		-------
		2-D narray
			array containing the beta values.
		"""
		return super().computeBetas(sequence)

	
	def splitTime(self,sequence: list) -> tuple:
		"""
		Given a trace it returns a sequence of observation and a sequence of
		waiting times. If the given trace is non-timed the output waiting time
		sequence is ``None``.

		Parameters
		----------
		sequence : list
			_description_

		Returns
		-------
		tuple
			_description_
		"""
		if type(sequence[1]) == float and type(sequence[0]) == str:
			times_seq = [sequence[i] for i in range(1,len(sequence),2)]
			obs_seq   = [sequence[i] for i in range(0,len(sequence),2)]
		else:
			times_seq = None
			obs_seq = sequence
		return (times_seq,obs_seq)

	def _processWork(self,sequence: list, times: int):
		times_seq, obs_seq = self.splitTime(sequence)
		if times_seq == None:
			timed = False
		else:
			timed = True
		alpha_matrix = self.computeAlphas(obs_seq, times_seq)
		beta_matrix  = self.computeBetas( obs_seq, times_seq)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq == 0.0:
			print(sequence)
			print("has prob 0")
			return False
		####################
		if self.update_constant:
			num_cste = array(self.h.transition_expr)
			den_cste = ones(len(num_cste))
			for itrans,trans in enumerate(self.h.transition_expr[1:]):
				if trans.is_real:
					s,ss = where(self.h.matrix == itrans+1)
					s,ss = s[0],ss[0]
					p = array([self.h_l(s,ss,o)*exp(-self.h_e(s)*t) for o,t in zip(obs_seq,times_seq)])
					num_cste[itrans+1] = dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],times/proba_seq).sum()
					den_cste[itrans+1] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*times_seq,times/proba_seq).sum()
		else:
			num_cste = None
			den_cste = None
			
		num_cat1 = zeros(len(self.parameters_cat[0]))
		den_cat1 = zeros(len(self.parameters_cat[0]))
		for iparam,param in enumerate(self.parameters_cat[0]):
			for s,ss in self.h.parameterIndexes(param):
				p = array([self.h_l(s,ss,o)*exp(-self.h_e(s)*t) for o,t in zip(obs_seq,times_seq)])
				num_cat1[iparam] += dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],times/proba_seq).sum()
				den_cat1[iparam] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*times_seq,self.c_pis[s,ss,iparam]*times/proba_seq).sum()

		num_cat2 = zeros(len(self.parameters_cat[1]))
		den_cat2 = zeros(len(self.parameters_cat[1]))
		for iparam,param in enumerate(self.parameters_cat[1]):
			for s,ss in self.h.parameterIndexes(param):
				p = array([self.h_l(s,ss,o)*exp(-self.h_e(s)*t) for o,t in zip(obs_seq,times_seq)])
				num_cat2[iparam] += dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],self.a_pis[s,ss,iparam]*times/proba_seq).sum()
				den_cat2[iparam] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*times_seq,self.a_pis[s,ss,self.h.parameter_str.index(param)]*self.hval[s,ss]*times/proba_seq).sum()
		
		terms_cat3 = []
		for iparam,param in enumerate(self.parameters_cat[2]):
			temp = [0.0]
			for s,ss in self.h.parameterIndexes(param):
				p = array([self.h_l(s,ss,o)*exp(-self.h_e(s)*t) for o,t in zip(obs_seq,times_seq)])
				temp[0] -= dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],self.a_pis[s,ss,self.h.parameter_str.index(param)]*times/proba_seq).sum()
				c = self.C(s,ss)
				for _ in range(1+c-len(temp)):
					temp.append(0.0)
				temp[c] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*times_seq,self.a_pis[s,ss,self.h.parameter_str.index(param)]*self.hval[s,ss]*times/proba_seq).sum()/(self.h.parameter_values[param]**c)
			terms_cat3.append(array(temp))

		return [den_cste, num_cste, den_cat1, num_cat1, den_cat2, num_cat2, terms_cat3, proba_seq, times]
			

	def _generateHhat(self,temp):
		den_cat1 = array([i[2] for i in temp]).sum(axis=0)
		num_cat1 = array([i[3] for i in temp]).sum(axis=0)
		den_cat2 = array([i[4] for i in temp]).sum(axis=0)
		num_cat2 = array([i[5] for i in temp]).sum(axis=0)
		terms_cat3 = [i[6] for i in temp]
		lst_proba=array([i[7] for i in temp])
		lst_times=array([i[8] for i in temp])

		currentloglikelihood = dot(log(lst_proba),lst_times)

		parameters = self.parameters_cat[0]+self.parameters_cat[1]+self.parameters_cat[2]
		values = []

		if self.update_constant:
			den_cste = array([i[0] for i in temp]).sum(axis=0)
			num_cste = array([i[1] for i in temp]).sum(axis=0)
			for p in range(len(den_cste)):
				if den_cste[p] == 0.0:
					den_cste[p] = 1.0
					num_cste[p] = self.h.transition_expr[p]
			self.h.transition_expr = [sympify(i) for i in (num_cste/den_cste).tolist()]
	
		for p in range(len(den_cat1)):
			if den_cat1[p] == 0.0:
				den_cat1[p] = 1.0
				num_cat1[p] = self.h.parameter_values[self.parameters_cat[0][p]]
		values += (num_cat1/den_cat1).tolist()

		for ip,p in enumerate(self.parameters_cat[1]):
			if den_cat2[ip] == 0.0:
				den_cat2[ip] = 1.0
				num_cat2[ip] = self.h.parameter_values[p]
			else:
				s,ss = self.h.parameterIndexes(p)[0]
				c = self.C(s,ss)
				den_cat2[ip] = den_cat2[ip]**(1/c)
				num_cat2[ip] = self.h.parameter_values[p]*num_cat2[ip]**(1/c)
		values += (num_cat2/den_cat2).tolist()

		for p in range(len(terms_cat3[0])):
			temp = array([i[p] for i in terms_cat3], dtype=float).sum(axis=0)
			values.append(max(polyroots(temp)))

		self.h.instantiate(parameters,values)
		return self.h, currentloglikelihood