from .PCTMC import *
from ..base.BW import BW
from ..base.Set import Set
from numpy import array, zeros, dot, append, ones, log, inf, newaxis, full


class BW_PCTMC(BW):
	def __init__(self) -> None:
		super().__init__()

	def fit(self, traces, initial_model: PCTMC, output_file: str=None,
			epsilon: float=0.01, max_it: int= inf, pp: str='',
			verbose: bool = True, return_data: bool = False,
			stormpy_output: bool = True, fixed_parameters: ndarray = False) -> PCTMC:
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
		fixed_parameters: ndarray of bool, optional
			ndarray of bool with the same shape as the transition matrix (i.e
			nb_states x nb_states). If `fixed_parameters[s1,s2] == True`, the
			transition parameter from s1 to s2 will not be changed during the
			learning (it's a fixed parameter).
			By default no parameters will be fixed.

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
				print("ERROR: the initial model is a Storm model and Storm is not installed on the machine")
				return False
			initial_model = stormpyModeltoJajapy(initial_model)	

		if not initial_model.isInstantiated():
			initial_model.randomInstantiation()
		
		self.nb_parameters = initial_model.nb_parameters
		#if type(fixed_parameters) == bool:
		#	self.fixed_parameters = full(initial_model.matrix.shape,False)
		#else:
		#	self.fixed_parameters = fixed_parameters

		return super().fit(traces, initial_model, output_file, epsilon, max_it, pp, verbose,return_data,stormpy_output)

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
		return self.h.e(s)
	
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
		return self.h.l(s1,s2,obs)

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
		alpha_matrix = append(init_arr,zero_arr).reshape(len_seq+1,self.nb_states)
		for k in range(len_seq):
			for s in range(self.nb_states):
				p = array([self.h_l(ss,s,obs_seq[k])*exp(-self.h_e(ss)*times_seq[k]) for ss in range(self.nb_states)])
				alpha_matrix[k+1,s] = dot(alpha_matrix[k],p)
		alpha_matrix[-1] *= (array(self.h.labeling) == obs_seq[-1])
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
			return False
		####################
		if timed:
			den = (alpha_matrix[:,:-1]*beta_matrix[:,:-1]*times_seq*times/proba_seq).sum(axis=1)	
		else:
			den = (alpha_matrix[:,:-1]*beta_matrix[:,:-1]*times/proba_seq).sum(axis=1)

		num_p = zeros(self.nb_parameters)
		den_p = zeros(self.nb_parameters)
		for p in range(self.nb_parameters):
			for s,ss in self.h.parameter_indexes[p]:
				if timed:
					p = array([self.h_l(s,ss,o)*exp(-self.h_e(s)*t) for o,t in zip(obs_seq,times_seq)])
		####################
		return [den, num, proba_seq, times]

	def _generateHhat(self,temp):
		den = array([i[0] for i in temp]).sum(axis=0)
		num = array([i[1] for i in temp]).sum(axis=0)
		lst_proba=array([i[2] for i in temp])
		lst_times=array([i[3] for i in temp])

		currentloglikelihood = dot(log(lst_proba),lst_times)

		for s in range(len(den)):
			if den[s] == 0.0:
				den[s] = 1.0
				num[s] = self.h.matrix[s]

		matrix = num/den[:, newaxis]
		matrix = self.h.matrix*self.fixed_parameters + matrix
		return [PCTMC(matrix,self.h.labeling),currentloglikelihood]