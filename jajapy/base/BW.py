from sys import platform
from multiprocessing import cpu_count, Pool
from numpy.polynomial.polynomial import polyroots
from numpy import array, dot, append, zeros, ones, float64, inf, ndarray, log, isnan, sqrt, stack, longdouble, float16, where, delete, newaxis, full, infty
from math import exp
from datetime import datetime
from .Set import Set
from alive_progress import alive_bar
from .Model import MC_ID, MDP_ID, CTMC_ID, PCTMC_ID, HMM_ID, GOHMM_ID
from sympy import sympify

NB_PROCESS = cpu_count()-1

class BW:
	"""
	Class for the Baum-Welch algorithm.
	"""
	def __init__(self):
		try:
			from ..with_stormpy import jajapyModeltoStormpy, stormpyModeltoJajapy
			self.jajapyModeltoStormpy = jajapyModeltoStormpy
			self.stormpyModeltoJajapy = stormpyModeltoJajapy
			self.stormpy_installed = True
		except ModuleNotFoundError:
			self.stormpy_installed = False


	def fit(self,training_set,initial_model=None, nb_states: int=None,
			random_initial_state: bool=True, min_exit_rate_time : int=None,
			max_exit_rate_time: int=None, self_loop: bool =None,
			nb_distributions: int=None, output_file: str=None,
			output_file_prism: str=None, epsilon: float=0.01, max_it: int=inf,
			pp: str='', verbose: bool=True, return_data: bool=False,
			stormpy_output: bool = True,fixed_parameters: ndarray = False,
			update_constant :bool = True, min_val: float = None, max_val: float = None):
		"""
		Fits any model according to ``traces``.
		This method will figure which type of Markov model should be used, according to the
		training set and the initial model (if given).

		Parameters
		----------
		traces : Set, list or ndarray.
			The training set.
		initial_model : Model or stormpy.sparse model, optional.
			The first hypothesis. If not set it will create a random Model
			with ``nb_states`` states.
		nb_states: int, optional.
			If ``initial_model`` is not set it will create a random Model with
			``nb_states`` states. Must be set if ``initial_model`` is not set.
		random_initial_state: bool, optional.
			If ``initial_model`` is not set it will create a random Model with
			random initial state according to this sequence of probabilities.
			Must be set if ``initial_model`` is not set.
			Default is True.
		min_exit_rate_time: int, optional
			For CTMC learning only.
			Minimum exit rate for the states in the first hypothesis.
			Must be set if ``initial_model`` is not set, and if the SUL is a
			CTMC or PCTMC.
		max_exit_rate_time: int, optional
			For CTMC learning only.
			Minimum exit rate for the states in the first hypothesis.
			Must be set if ``initial_model`` is not set, and if the SUL is a
			CTMC or PCTMC.
		self_loop: bool, optional
			For CTMC/PCTMC learning only.
			Wether or not there will be self loop in the first hypothesis.
			Must be set if ``initial_model`` is not set, and if the SUL is a
			CTMC or PCTMC.
		nb_distributions: int, optional.
			For GoHMM learning only.
			Number of distributions in each state in the initial hypothsis.
			Must be set if ``initial_model`` is not set, and if the SUL is a
			GoHMM.
		output_file : str, optional
			If set, the output model will be saved at this location.
			Otherwise the output model will not be saved.
		output_file_prism : str, optional
			If set, the output model will be saved in a prism file at this
			location. Otherwise the output model will not be saved.
			This parameter is ignored if the model under learning is a HMM
			or a GoHMM.
		epsilon : float, optional
			The learning process stops when the difference between the
			loglikelihood of the training set under the two last hypothesis is
			lower than ``epsilon``. The lower this value the better the output,
			but the longer the running time.
			Default is 0.01.
		max_it: int, optional
			Maximal number of iterations. The algorithm will stop after `max_it`
			iterations.
			Default is infinity.
		pp:	str, optional
			Will be printed at each iteration.
			Default is an empty string.
		verbose: bool, optional.
			Print or not a small recap at the end of the learning.
			Default is True.
		return_data: bool, optional.
			If set to True, a dictionary containing following values will be
			returned alongside the output model once the learning is done.
			'learning_rounds', 'learning_time', 'training_set_loglikelihood'.
			Default is False.
		stormpy_output: bool, optional.
			If set to True the output model will be a Stormpy sparse model.
			Doesn't work for HMM and GOHMM.
			Default is True.
		fixed_parameters: ndarray of bool, optional
			For CTMC/PCTMC learning only.
			ndarray of bool with the same shape as the transition matrix (i.e
			nb_states x nb_states). If `fixed_parameters[s1,s2] == True`, the
			transition parameter from s1 to s2 will not be changed during the
			learning (it's a fixed parameter).
			By default no parameters will be fixed.
		update_constant: bool, optional
			For PCTMC learning only.
			If set to False, the constant transitions (i.e. tha transition
			that doesn't depend on any parameter) will no be updated.
			Default is True.
		min_val: float, optional
			For PCTMC learning only.
			Minimal value for the randomly instantiated parameters.
			If not set and if the model has at least two instantiated parameters,
			this value is equal to the parameters with the smallest instantiation.
			If not set and if the model has less than two instantiated parameters,
			this value is equal to 0.1.
		max_val : float, optional
			For PCTMC learning only.
			Maximal value for the randomly instantiated parameters.
			If not set and if the model has at least two instantiated parameters,
			this value is equal to the parameters with the highest instantiation.
			If not set and if the model has less than two instantiated parameters,
			this value is equal to 5.0.	

		Returns
		-------
		Model
			fitted model.
		"""

		stormpy_output = self._preparation(training_set,initial_model, nb_states,
						  random_initial_state, min_exit_rate_time,
						  max_exit_rate_time, self_loop,
						  nb_distributions, stormpy_output, 
						  fixed_parameters, update_constant,
						  min_val, max_val)
		return self._bw( max_it,pp,epsilon,output_file,output_file_prism,verbose,
						stormpy_output,return_data)


	def _preparation(self,training_set,initial_model=None, nb_states: int=None,
			random_initial_state: bool=True, min_exit_rate_time : int=None,
			max_exit_rate_time: int=None, self_loop: bool =None,
			nb_distributions: int=None,
			stormpy_output: bool = True,fixed_parameters: ndarray = False,
			update_constant :bool = True, min_val: float = None, max_val: float = None):	

		self._getInitialModel(initial_model,nb_states)
		self._getTrainingSet(training_set)
		if self.h == None:
			self._createInitialModel(nb_states, random_initial_state,
								 min_exit_rate_time, max_exit_rate_time, self_loop,
								 nb_distributions)
		else:
			if ((self.training_set.type != self.type_model) and
			   (self.training_set.type != 0 and self.type_model != 2) and
			   (self.training_set.type != 0 and self.type_model != 4) and
			   (self.training_set.type != 0 and self.type_model != 5) and
			   (self.training_set.type != 4 and self.type_model != 5)):
				msg = "the training set and the initial hypothesis are notcompatible:\n"
				msg+= "the training set is a set of sequences of "
				msg+=['labels','action-label pairs','vector of continuous observations','dwell time-label pairs'][self.training_set.type]
				msg+= "\nand the initial hypothesis is a "+['MC','MDP','HMM','GoHMM','CTMC','PCTMC'][self.type_model]
				raise RuntimeError(msg)
		
		if self.type_model == MC_ID:
			self._update = self._update_MC
			self._processWork  = self._processWork_MC
			print("Learning an MC...")
		
		elif self.type_model == MDP_ID:
			self._h_tau = self._h_tau_MDP
			self._computeAlphas = self._computeAlphas_MDP
			self._computeBetas = self._computeBetas_MDP
			self._update = self._update_MDP
			self._processWork  = self._processWork_MDP
			print("Learning an MDP...")
			self.actions = self.h.actions
		
		elif self.type_model == CTMC_ID:
			self._update = self._update_CTMC
			self._processWork = self._processWork_CTMC
			print("Learning a CTMC...")
			if self.training_set.type == 4:
				self._computeAlphas = self._computeAlphas_timed
				self._computeBetas  = self._computeBetas_timed
			if type(fixed_parameters) == bool:
				self.fixed_parameters = full(self.h.matrix.shape,False)
			else:
				self.fixed_parameters = fixed_parameters
		
		elif self.type_model == PCTMC_ID:
			print("Learning a PCTMC...")
			self._update = self._update_PCTMC
			self._processWork = self._processWork_PCTMC
			self._computeTaus = self._computeTaus_PCTMC
			if min_val != None:
				if max_val != None:
					self.h.randomInstantiation(min_val=min_val, max_val=max_val)
				else:
					self.h.randomInstantiation(min_val=min_val)
			elif max_val != None:
				self.h.randomInstantiation(max_val=max_val)
			else:
				self.h.randomInstantiation()
				
			if self.training_set.type == 4:
				self._computeAlphas = self._computeAlphas_timed
				self._computeBetas  = self._computeBetas_timed
			self.nb_parameters = self.h.nb_parameters
			self.update_constant = update_constant
			self._h_e = self._h_e_PCTMC
			self._h_l = self._h_l_PCTMC
			self._h_tau=self._h_tau_PCTMC
			self._sortParameters(fixed_parameters)

		elif self.type_model == HMM_ID:
			self._update = self._update_HMM
			self._processWork = self._processWork_HMM
			self.alphabet = self.h.getAlphabet()
			print("Learning an HMM...")
		
		elif self.type_model == GOHMM_ID:
			self._update= self._update_GoHMM
			self._processWork = self._processWork_GoHMM
			self.nb_distr = self.h.nb_distributions
			print("Learning a GoHMM...")

		if stormpy_output and not self.stormpy_installed:
			print("WARNING: stormpy not found. The output model will be a Jajapy model")
			stormpy_output = False
		
		return stormpy_output

	def _bw(self,max_it,pp,epsilon,output_file,output_file_prism,verbose,stormpy_output,return_data):
		start_time = datetime.now()
		self.nb_states = self.h.nb_states

		counter = 0
		prevloglikelihood = -infty

		if max_it != inf:
			alive_parameter = max_it
		else:
			alive_parameter = 0
		
		with alive_bar(alive_parameter, dual_line=True, title=str(pp)) as bar:
			while counter < max_it:
				self._computeTaus()
				temp = self._runProcesses(self.training_set)
				currentloglikelihood = self._update(temp)
				counter += 1
				bar_text = '   Diff. loglikelihood: '+str(round(currentloglikelihood-prevloglikelihood,5))+' (>'+str(epsilon)+')'
				bar_text+= '   Av. one iteration (s): '+str(round((datetime.now()-start_time).total_seconds()/counter,2))
				bar.text = bar_text
				bar()
				if currentloglikelihood - prevloglikelihood < epsilon:
					break
				else:
					prevloglikelihood = currentloglikelihood

		running_time = datetime.now()-start_time
		running_time = running_time.total_seconds()

		if output_file:
			self.h.save(output_file)
		
		if output_file_prism and (self.type_model != HMM_ID and self.type_model != GOHMM_ID):
			self.h.savePrism(output_file_prism)
		
		if verbose:
			self._endPrint(counter,running_time)

		if stormpy_output and (self.type_model != HMM_ID and self.type_model != GOHMM_ID):
			self.h = self.jajapyModeltoStormpy(self.h)

		if return_data:
			info = {"learning_rounds":counter,"learning_time":running_time,"training_set_loglikelihood":currentloglikelihood}
			return self.h, info
			
		return self.h
	

	# MC-----------------------------------------------------------------------
	def _processWork_MC(self,sequence,times):
		alpha_matrix = self._computeAlphas(sequence)
		beta_matrix  = self._computeBetas( sequence)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq == 0.0:
			return False
		####################
		den = (alpha_matrix.T[:-1]*beta_matrix.T[:-1]*times/proba_seq).sum(axis=0)			
		num = zeros(shape=(self.nb_states,self.nb_states))
		for s in range(self.nb_states):
			for ss in range(self.nb_states):
				p = array([self._h_tau(s,ss,o) for o in sequence])
				num[s,ss] = dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],times/proba_seq).sum()
		####################
		return [den,num, proba_seq,times]
	
	def _update_MC(self,temp):
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
		self.h.matrix = matrix
		return currentloglikelihood


	# MDP----------------------------------------------------------------------
	def _h_tau_MDP(self,s1: int,act: str,s2: int,obs: str) -> float:
		"""
		Returns the probability of moving from state ``s1`` executing `action`
		to ``s2`` generating observation ``obs``.

		Parameters
		----------
		s1: int
			source state ID.
		action: str
			An action.
		s2: int
			destination state ID.
		obs: str
			generated observation.
		
		Returns
		-------
		float
			A probability.
		"""
		return self.h.tau(s1,act,s2,obs)

	def _computeAlphas_MDP(self,sequence: list, sequence_actions: list) -> array:
		"""
		Compute the alpha values for ``sequence`` under the current BW
		hypothesis.

		Parameters
		----------
		sequence : list of str
			sequence of observations.
		sequence_actions : list of str
			sequence of actions.

		Returns
		-------
		2-D narray
			array containing the alpha values.
		"""
		len_seq = len(sequence)
		init_arr = self.h.initial_state
		zero_arr = zeros(shape=((len_seq-1)*self.nb_states,))
		alpha_matrix = append(init_arr,zero_arr).reshape(len_seq,self.nb_states)
		for k in range(len_seq-1):
			for s in range(self.nb_states):
				p = array([self._h_tau(ss,sequence_actions[k],s,sequence[k]) for ss in range(self.nb_states)])
				alpha_matrix[k+1,s] = dot(alpha_matrix[k],p)
		
		alpha_matrix[-1] *= (array(self.h.labelling) == sequence[-1])
		return alpha_matrix.T

	def _computeBetas_MDP(self,sequence: list,sequence_actions: list) -> array:
		"""
		Compute the beta values for ``sequence`` under the current BW
		hypothesis.

		Parameters
		----------
		sequence : list of str
			sequence of observations.
		sequence_actions : list of str
			sequence of actions.

		Returns
		-------
		2-D narray
			array containing the beta values.
		"""
		len_seq = len(sequence)
		init_arr = ones(self.nb_states)*(array(self.h.labelling) == sequence[-1])
		zero_arr = zeros(shape=((len_seq-1)*self.nb_states,))
		beta_matrix = append(zero_arr,init_arr).reshape(len_seq,self.nb_states)
		for k in range(len(sequence)-2,-1,-1):
			for s in range(self.nb_states):
				p = array([self._h_tau(s,sequence_actions[k],ss,sequence[k]) for ss in range(self.nb_states)])
				beta_matrix[k,s] = dot(beta_matrix[k+1],p)
		return beta_matrix.T

	def _processWork_MDP(self,sequence,times):
		sequence_actions = [sequence[i+1] for i in range(0,len(sequence)-1,2)]
		sequence_obs = [sequence[i] for i in range(0,len(sequence)-1,2)]+[sequence[-1]]
		alpha_matrix = self._computeAlphas(sequence_obs,sequence_actions)
		beta_matrix = self._computeBetas(sequence_obs,sequence_actions)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq != 0.0:
			den = zeros(shape=(self.nb_states,len(self.actions)))
			num = zeros(shape=(self.nb_states,len(self.actions),self.nb_states))
			
			for s in range(self.nb_states):
				for i,a in enumerate(self.actions):
					arr_dirak = array([1.0 if t == a else 0.0 for t in sequence_actions])
					den[s,i] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*arr_dirak,times/proba_seq).sum()
			
				for ss in range(self.nb_states):
					p = array([self._h_tau(s,a,ss,o) for o,a in zip(sequence_obs[:-1],sequence_actions)])
					for ia,act in enumerate(self.actions):
						arr_dirak = [1.0 if a == act else 0.0 for a in sequence_actions]
						num[s,ia,ss] = dot(alpha_matrix[s][:-1]*arr_dirak*beta_matrix[ss][1:]*p,times/proba_seq).sum()
			return [den,num,proba_seq,times]
		return False

	def _update_MDP(self,temp):
		den = array([i[0] for i in temp]).sum(axis=0)
		if type(den) == float64: # den == 0.0
			msg = 'The current hypothesis is not able to generate any of the '
			msg+= 'sequence in the training set.'
			raise ValueError(msg)
		
		num = array([i[1] for i in temp]).sum(axis=0)
		lst_proba=array([i[2] for i in temp])
		lst_times=array([i[3] for i in temp])

		currentloglikelihood = dot(log(lst_proba),lst_times)

		for s in range(self.nb_states):
			for a in range(len(self.actions)):
				if den[s][a] == 0.0 or isnan(den[s][a]):
					den[s][a] = 1.0
					num[s][a] = self.h.matrix[s][a]
		den = den.reshape(self.nb_states,len(self.actions),1)
		matrix = num/den
		self.h.matrix = matrix
		return currentloglikelihood


	# CTMC---------------------------------------------------------------------
	def _h_e(self,s: int) -> float:
		"""
		Returns the exit rate, in the current hypothesis, of state ``s``, i.e.
		the sum of all the rates in this state.

		Parameters
		----------
		s : int
			A state ID.
		
		Returns
		-------
		float
			An exit rate.
		"""
		return self.h.e(s)
	
	def _h_l(self, s1: int, s2: int, obs: str) -> float:
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
	
	def _computeAlphas_timed(self,obs_seq: list, times_seq: list) -> array:
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
		alpha_matrix = append(init_arr,zero_arr).reshape(len_seq+1,self.nb_states).astype(longdouble)
		for k in range(len_seq):
			for s in range(self.nb_states):
				p = array([self._h_l(ss,s,obs_seq[k])*exp(-self._h_e(ss)*times_seq[k]) for ss in range(self.nb_states)])
				alpha_matrix[k+1,s] = dot(alpha_matrix[k],p)
		alpha_matrix[-1] *= (array(self.h.labelling) == obs_seq[-1])
		return alpha_matrix.T

	def _computeBetas_timed(self,obs_seq: list, times_seq: list) -> array:
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
		init_arr = ones(self.nb_states)*(array(self.h.labelling) == obs_seq[-1])
		zero_arr = zeros(shape=(len_seq*self.nb_states,))
		beta_matrix = append(zero_arr,init_arr).reshape(len_seq+1,self.nb_states).astype(longdouble)
		for k in range(len_seq-1,-1,-1):
			for s in range(self.nb_states):
				p = array([self._h_l(s,ss,obs_seq[k]) for ss in range(self.nb_states)])
				p = p * exp(-self._h_e(s)*times_seq[k])
				beta_matrix[k,s] = dot(beta_matrix[k+1],p)
		return beta_matrix.T

	def _splitTime(self,sequence: list) -> tuple:
		"""
		Given a trace it returns a sequence of observation and a sequence of
		waiting times. If the given trace is non-timed the output waiting time
		sequence is ``None``.

		Parameters
		----------
		sequence : list
			the trace to split.

		Returns
		-------
		tuple
			a tuple containing a sequence of observations and a sequence of
			waiting times.
		"""
		if type(sequence[1]) == float and type(sequence[0]) == str:
			times_seq = [sequence[i] for i in range(1,len(sequence),2)]
			obs_seq   = [sequence[i] for i in range(0,len(sequence),2)]
		else:
			times_seq = None
			obs_seq = sequence
		return (times_seq,obs_seq)

	def _processWork_CTMC(self,sequence: list, times: int):
		times_seq, obs_seq = self._splitTime(sequence)
		if times_seq == None:
			timed = False
		else:
			timed = True
		alpha_matrix = self._computeAlphas(obs_seq, times_seq)
		beta_matrix  = self._computeBetas( obs_seq, times_seq)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq == 0.0:
			return False
		####################
		if timed:
			den = (alpha_matrix[:,:-1]*beta_matrix[:,:-1]*times_seq*times/proba_seq).sum(axis=1)	
		else:
			den = (alpha_matrix[:,:-1]*beta_matrix[:,:-1]*times/proba_seq).sum(axis=1)

		num = zeros(shape=(self.nb_states,self.nb_states))	
		for s in range(self.nb_states):
			for ss in range(self.nb_states):
				if not self.fixed_parameters[s,ss]:
					if timed:
						p = array([self._h_l(s,ss,o)*exp(-self._h_e(s)*t) for o,t in zip(obs_seq,times_seq)])
					else:
						p = array([self._h_l(s,ss,o)/self._h_e(s) for o in obs_seq]) # not sure
					num[s,ss] = dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],times/proba_seq).sum()
				else:
					num[s,ss] = 0.0
		####################
		return [den, num, proba_seq, times]

	def _update_CTMC(self,temp):
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

		self.h.matrix = matrix
		return currentloglikelihood


	# HMM----------------------------------------------------------------------
	def _processWork_HMM(self,sequence,times):
		alpha_matrix = self._computeAlphas(sequence)
		beta_matrix = self._computeBetas(sequence)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq != 0.0:
			####################
			den = (alpha_matrix.T[:-1]*beta_matrix.T[:-1]*times/proba_seq).sum(axis=0)
			####################
			num_a = zeros(shape=(self.nb_states,self.nb_states))
			num_b = zeros(shape=(self.nb_states,len(self.alphabet)))
			for s in range(self.nb_states):
				for ss in range(self.nb_states):
					p = array([self._h_tau(s,ss,o) for o in sequence])
					num_a[s,ss] = dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],times/proba_seq).sum()
					temp = alpha_matrix[s][:-1]*beta_matrix[s][:-1]*times/proba_seq
					for o, obs in enumerate(self.alphabet):
						arr_dirak = [1.0 if t == obs else 0.0 for t in sequence]
						num_b[s,o] = (temp*arr_dirak).sum()
			####################
			num_init = alpha_matrix.T[0]*beta_matrix.T[0]*times/proba_seq
			####################
			return [den, num_a, num_b, proba_seq, times, num_init]
		return False

	def _update_HMM(self,temp):
		den = array([i[0] for i in temp]).sum(axis=0)
		num_a = array([i[1] for i in temp]).sum(axis=0)
		num_b = array([i[2] for i in temp]).sum(axis=0)
		lst_proba=array([i[3] for i in temp])
		lst_times=array([i[4] for i in temp])
		lst_init =array([i[5] for i in temp]).T
		
		currentloglikelihood = dot(log(lst_proba),lst_times)

		for s in range(len(den)):
			if den[s] == 0.0:
				den[s] = 1.0
				num_a[s] = self.h.matrix[s]
				num_b[s] = self.h.output[s]

		matrix = num_a/den[:, newaxis]
		output = num_b/den[:, newaxis]

		initial_state = array([lst_init[s].sum()/lst_init.sum() for s in range(self.nb_states)])


		self.h.matrix = matrix
		self.h.output = output
		self.h.initial_state = initial_state
		return currentloglikelihood

	# GoHMM--------------------------------------------------------------------
	def _processWork_GoHMM(self,sequence,times):
		sequence = array(sequence)
		alpha_matrix = self._computeAlphas(sequence,longdouble)
		if isnan(alpha_matrix).any():
			return False
		beta_matrix = self._computeBetas(sequence,longdouble)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq != 0.0:
			den = (alpha_matrix.T[:-1]*beta_matrix.T[:-1]*times/proba_seq).sum(axis=0)
			num_a  = zeros(shape=(self.nb_states,self.nb_states))
			num_mu = zeros(shape=(self.nb_states,self.nb_distr))
			num_va = zeros(shape=(self.nb_states,self.nb_distr))
			for s in range(self.nb_states):
				num_mu[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*sequence.T,times/proba_seq).sum(axis=1)
				num_va[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*(sequence-self.h.mu(s)).T**2,times/proba_seq).sum(axis=1)
				for ss in range(self.nb_states):
					p = array([self._h_tau(s,ss,o) for o in sequence])
					num_a[s,ss] = dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],times/proba_seq).sum()
			num_init = alpha_matrix.T[0]*beta_matrix.T[0]*times/proba_seq
			return [den,num_a,num_mu,num_va,proba_seq,times,num_init]
		return False

	def _update_GoHMM(self,temp):
		a  = array([i[1] for i in temp]).sum(axis=0)
		den= array([i[0] for i in temp]).sum(axis=0)
		mu = array([i[2] for i in temp]).sum(axis=0)
		va = array([i[3] for i in temp]).sum(axis=0)
		lst_proba=array([i[4] for i in temp])
		lst_times=array([i[5] for i in temp])
		init =array([i[6] for i in temp]).sum(axis=0)

		currentloglikelihood = dot(log(lst_proba),lst_times)

		output_mu = (mu.T/den).T
		output_va = sqrt((va.T/den).T)

		matrix = a/den[:, newaxis]
		output = stack((output_mu,output_va),axis=2)

		initial_state = array([init[s]/init.sum() for s in range(self.nb_states)])
		
		self.h.matrix = matrix
		self.h.output = output
		self.h.initial_state = initial_state
		return currentloglikelihood


	# PCTMC--------------------------------------------------------------------
	def _sortParameters(self,fixed_parameters: list):
		"""
		Sort the parameters into the three categories.
		Depending on the category, the parameters are estimated differently:
		in some cases some optimizations are possible (see the paper).
		
		Parameters
		----------
		fixed_parameters : list
			list of all fixed parameters (the one we are not estimating).
		"""
		self.a_pis = zeros((self.h.nb_states,self.h.nb_states,len(self.h.parameter_str)),dtype=float16)
		for iparam,param in enumerate(self.h.parameter_str):
			if not param in fixed_parameters:
				for s,ss in self.h.parameterIndexes(param):
					self.a_pis[s,ss,iparam] = self._a_pi(s,ss,param)

		self.parameters_cat = [[],[],[]]
		for ip,p in enumerate(self.h.parameter_str):
			if not p in fixed_parameters:
				apis = [self.a_pis[x,y,ip] for x,y in self.h.parameterIndexes(p)]
				cs = [self._C(x,y) for x,y in self.h.parameterIndexes(p)]
				if min(apis) == 1 and max(apis) == 1 and min(cs) == 1 and max(cs) == 1:
					self.parameters_cat[0].append(p)
				elif min(cs) == max(cs):
					self.parameters_cat[1].append(p)
				else:
					self.parameters_cat[2].append(p)
		
		self.c_pis = zeros((self.h.nb_states,self.h.nb_states))
		for iparam,param in enumerate(self.parameters_cat[0]):
			for s,ss in self.h.parameterIndexes(param):
				self.c_pis[s,ss] = self._c_pi(s,ss,param)

	def _computeTaus_PCTMC(self):
		self.hval = zeros((self.nb_states,self.nb_states))
		for s in range(self.nb_states):
			for ss in range(self.nb_states):
				self.hval[s,ss] =  self.h.transitionValue(s,ss)

	def _a_pi(self,s1: int,s2: int, p: str) -> int:
		"""
		Return the power of parameter `p` in the transition from `s1` to `s2`.

		Parameters
		----------
		s1 : int
			source state ID.
		s2 : int
			destination state ID.
		p : str
			parameter name.

		Returns
		-------
		int
			Return the power of parameter `p` in the transition from `s1` to `s2`.
		"""
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

	def _c_pi(self,s1: int,s2: int, p:str) -> float:
		"""
		Return the coefficient of parameter `p` in the transition from
		`s1` to `s2`.

		Parameters
		----------
		s1 : int
			source state ID.
		s2 : int
			destination state ID.
		p : str
			parameter name.

		Returns
		-------
		float
			the coefficient of parameter `p` in the transition from
			`s1` to `s2`.
		"""
		#used only if p is the only non-fixed parameter
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
			res = 1.0
			for a in t.args:
				if a.is_Symbol:
					if not a.name == p:
						res *= self.h.parameterValue(a.name)
				else:
					res *= a
			return float(res)
		else:
			return 1

	def _C(self,s1:int ,s2:int) -> int:
		"""
		Returns the sum of the power of all parameter in the transition
		from `s1` to `s2`.

		Parameters
		----------
		s1 : int
			source state ID.
		s2 : int
			destination state ID.

		Returns
		-------
		int
			the sum of the power of all parameter in the transition
		"""
		r = 0
		for p in self.h.involvedParameters(s1,s2):
			r += self.a_pis[s1,s2,self.h.parameter_str.index(p)]
		return int(r)

	def _h_e_PCTMC(self,s: int) -> float:
		"""
		Returns the exit rate, in the current hypothesis, of state ``s``, i.e.
		the sum of all the rates in this state.

		Parameters
		----------
		s : int
			A state ID.

		Returns
		-------
		float
			An exit rate.
		"""
		return self.hval[s].sum()

	def _h_l_PCTMC(self, s1: int, s2: int, obs: str) -> float:
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
		if self.h.labelling[s1] != obs:
			return 0.0
		return self.hval[s1,s2]

	def _h_tau_PCTMC(self,s1: int,s2: int,obs: str) -> float:
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
		if self.h.labelling[s1] != obs:
			return 0.0
		e = self._h_e(s1)
		if e == 0.0:
			return inf
		return self.hval[s1,s2]/e

	def _processWork_PCTMC(self,sequence: list, times: int):
		times_seq, obs_seq = self._splitTime(sequence)
		if times_seq == None:
			timed = False
		else:
			timed = True
		alpha_matrix = self._computeAlphas(obs_seq, times_seq)
		beta_matrix  = self._computeBetas( obs_seq, times_seq)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq == 0.0:
			return False
		####################
		if self.update_constant:
			num_cste = array(self.h.transition_expr)
			den_cste = ones(len(num_cste))
			for itrans,trans in enumerate(self.h.transition_expr[1:]):
				if trans.is_real:
					s,ss = where(self.h.matrix == itrans+1)
					s,ss = s[0],ss[0]
					if timed:
						p = array([self._h_l(s,ss,o)*exp(-self._h_e(s)*t) for o,t in zip(obs_seq,times_seq)])
						den_cste[itrans+1] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*times_seq,times/proba_seq).sum()
					else:
						p = array([self._h_l(s,ss,o)/self._h_e(s) for o in obs_seq])
						den_cste[itrans+1] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1],times/proba_seq).sum()
					num_cste[itrans+1] = dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],times/proba_seq).sum()

		else:
			num_cste = None
			den_cste = None

		#print(sequence, proba_seq)
		num_cat1 = zeros(len(self.parameters_cat[0]))
		den_cat1 = zeros(len(self.parameters_cat[0]))
		for iparam,param in enumerate(self.parameters_cat[0]):
			for s,ss in self.h.parameterIndexes(param):
				if timed:
					p = array([self._h_l(s,ss,o)*exp(-self._h_e(s)*t) for o,t in zip(obs_seq,times_seq)])
				else:
					p = array([self._h_l(s,ss,o)/self._h_e(s) for o in obs_seq])

				if p.sum()>0.0:
					if timed:
						den_cat1[iparam] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*times_seq,self.c_pis[s,ss]*times/proba_seq).sum()
					else:
						den_cat1[iparam] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1],self.c_pis[s,ss]*times/(self._h_e(s)*proba_seq)).sum()
					num_cat1[iparam] += dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],times/proba_seq).sum()

		num_cat2 = zeros(len(self.parameters_cat[1]))
		den_cat2 = zeros(len(self.parameters_cat[1]))
		for iparam,param in enumerate(self.parameters_cat[1]):
			p_index = self.h.parameter_str.index(param)
			for s,ss in self.h.parameterIndexes(param):
				if timed:
					p = array([self._h_l(s,ss,o)*exp(-self._h_e(s)*t) for o,t in zip(obs_seq,times_seq)])
					den_cat2[iparam] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*times_seq,self.a_pis[s,ss,p_index]*self.hval[s,ss]*times/proba_seq).sum()
				else:
					p = array([self._h_l(s,ss,o)/self._h_e(s) for o in obs_seq])
					den_cat2[iparam] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1],self.a_pis[s,ss,p_index]*self.hval[s,ss]*times/(proba_seq*self._h_e(s))).sum()
				num_cat2[iparam] += dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],self.a_pis[s,ss,p_index]*times/proba_seq).sum()

		terms_cat3 = []
		for iparam,param in enumerate(self.parameters_cat[2]):
			p_index = self.h.parameter_str.index(param)
			temp = [0.0]
			for s,ss in self.h.parameterIndexes(param):
				if timed:
					p = array([self._h_l(s,ss,o)*exp(-self._h_e(s)*t) for o,t in zip(obs_seq,times_seq)])
				else:
					p = array([self._h_l(s,ss,o)/self._h_e(s) for o in obs_seq])
				temp[0] -= dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],self.a_pis[s,ss,p_index]*times/proba_seq).sum()
				c = self._C(s,ss)
				for _ in range(1+c-len(temp)):
					temp.append(0.0)
				if timed:
					temp[c] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*times_seq,self.a_pis[s,ss,p_index]*self.hval[s,ss]*times/proba_seq).sum()/(self.h.parameter_values[param]**c)
				else:
					temp[c] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1],self.a_pis[s,ss,p_index]*self.hval[s,ss]*times/proba_seq).sum()/(self._h_e(s)*self.h.parameter_values[param]**c)

			terms_cat3.append(array(temp))

		return [den_cste, num_cste, den_cat1, num_cat1, den_cat2, num_cat2, terms_cat3, proba_seq, times]

	def _update_PCTMC(self,temp):
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
				c = self._C(s,ss)
				den_cat2[ip] = den_cat2[ip]**(1/c)
				num_cat2[ip] = self.h.parameter_values[p]*num_cat2[ip]**(1/c)
		values += (num_cat2/den_cat2).tolist()

		for p in range(len(terms_cat3[0])):
			temp = array([i[p] for i in terms_cat3], dtype=float).sum(axis=0)
			values.append(max(polyroots(temp)))

		self.h.instantiate(parameters,values)
		return currentloglikelihood

	def fit_nonInstantiatedParameters(self, traces, initial_model,
			epsilon: float=0.01, max_it: int= inf,
			pp: str='', verbose: bool = True, return_data: bool = False,
			min_val: float = None, max_val: float = None) -> dict:
		"""
		For PCTMC learning only.
		Fits only the non-instantiated parameters in the initial model
		according to ``traces``.

		Parameters
		----------
		traces : Set or list or numpy.ndarray
			training set.
		initial_model : PCTMC
			first hypothesis.
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
		min_val: float, optional
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

		Returns
		-------
		dict or list
			Dictionary containing the estimated values for the non-indtantiated
			parameters.
			If `return_data` is set to True, returns a list containing:
			- the dictionary described above,
			- the `returned_data` (see parameter description).
		"""
		if initial_model.model_type != PCTMC_ID:
			raise RuntimeError("The initial model must be a PCTMC")
		to_update = []
		for p in initial_model.parameter_str:
			to_update.append(p)
		
		return self.fit_parameters(traces,initial_model,to_update,epsilon,
								max_it,pp,verbose,return_data,min_val,max_val)

	def fit_parameters(self, traces, initial_model, to_update: list,
			epsilon: float=0.01, max_it: int= inf,
			pp: str='', verbose: bool = True, return_data: bool = False,
			min_val: float = None, max_val: float = None) -> dict:
		"""
		For PCTMC learning only.
		Fits only the non-instantiated parameters in the initial model
		according to ``traces``.

		Parameters
		----------
		traces : Set or list or numpy.ndarray
			training set.
		initial_model : PCTMC
			first hypothesis.
		to_update: list of str
			list of parameter names to update
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
		min_val: float, optional
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

		Returns
		-------
		dict or list
			Dictionary containing the estimated values for the non-indtantiated
			parameters.
			If `return_data` is set to True, returns a list containing:
			- the dictionary described above,
			- the `returned_data` (see parameter description).
		"""
		self.type_model = PCTMC_ID
		if initial_model.model_type != PCTMC_ID:
			raise RuntimeError("The initial model must be a PCTMC")
		self.h = initial_model

		update_constant = False
		fixed_parameters = []
		
		for p in initial_model.parameter_str:
			if not p in to_update:
				fixed_parameters.append(p)
		for p in to_update:
			if not p in initial_model.parameter_str:
				raise RuntimeError('Parameter '+p+' is not in the initial model.')

		print("Fitting only parameters: ",end='')
		print(', '.join(to_update))

		# remove all the states never used in the training set:
		# we only care here about the values of some parameters,
		# the structure of the model is out of scope.
		if type(traces) != Set:
			traces = Set(traces)
		alphabet = traces.getAlphabet()
		print("Removing unused states: ", end='')
		print('from',initial_model.nb_states,'states to',end=' ')
		s = 0
		while s < initial_model.nb_states:
			if initial_model.getLabel(s) not in alphabet:
				initial_model.nb_states -= 1
				initial_model.labelling = initial_model.labelling[:s]+initial_model.labelling[s+1:]
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
		#initial_model.alphabet = alphabet
		print(initial_model.nb_states)

		self._getTrainingSet(traces)
		
		if min_val != None:
			if max_val != None:
				self.h.randomInstantiation(min_val=min_val, max_val=max_val)
			else:
				self.h.randomInstantiation(min_val=min_val)
		elif max_val != None:
			self.h.randomInstantiation(max_val=max_val)
		
		if self.training_set.type == 4:
			self._computeAlphas = self._computeAlphas_timed
			self._computeBetas  = self._computeBetas_timed
		self.nb_parameters = self.h.nb_parameters
		self.update_constant = update_constant
		self._processWork = self._processWork_PCTMC
		self._update = self._update_PCTMC
		self._computeTaus = self._computeTaus_PCTMC
		self._h_e = self._h_e_PCTMC
		self._h_l = self._h_l_PCTMC
		self._h_tau=self._h_tau_PCTMC
		self._sortParameters(fixed_parameters)

		returned = self._bw(max_it,pp,epsilon,False,False,verbose,False,return_data)
		
		res = {}
		for p in to_update:
			res[p] = self.h.parameterValue(p)
		if return_data:
			return res, returned[1]
		else:
			return res


	# General------------------------------------------------------------------
	def _getInitialModel(self,initial_model, nb_states: int):

		if initial_model == None:
			if nb_states == None:
				raise RuntimeError("initial_state or the nb_states must be given. Here none of them is.")
			else:
				self.h = None
				self.type_model = None
				return	
		try:
			initial_model.name
		except AttributeError: # then initial_model is a stormpy sparse model
			if not self.stormpy_installed:
				raise RuntimeError("the initial model is a Storm model and Storm is not installed on the machine")
			initial_model = self.stormpyModeltoJajapy(initial_model)
		self.h = initial_model
		self.type_model = initial_model.model_type
			
	def _getTrainingSet(self,training_set):
		if type(training_set) == ndarray:
			training_set = training_set.tolist()
		if type(training_set) == list:
			training_set = Set(training_set)
			if training_set.type == 0:
				a = set()
				l = set()
				mdp = True
				for seq in training_set.sequences:
					a = a.union(set([seq[k] for k in range(1,len(seq),2)]))
					l = l.union(set([seq[k] for k in range(0,len(seq),2)]))
					if a.intersection(l) != {}:
						mdp = False
						break
				if mdp:
					training_set.type = MDP_ID
		if training_set.type != 3 and self.type_model != HMM_ID:
			if training_set.type == 1:
				_, self.alphabet = training_set.getActionsObservations()
			else:
				self.alphabet = training_set.getAlphabet()
			if not 'init' in self.alphabet:
				for s in range(len(training_set.sequences)):
					training_set.sequences[s].insert(0,'init')
			else:
				self.alphabet.remove('init')

		self.training_set = training_set

	def _createInitialModel(self,nb_states, random_initial_state,
							min_exit_rate_time, max_exit_rate_time, self_loop,
							nb_distributions):
		if self.training_set.type == 0:
			from ..mc.MC import MC_random
			self.type_model = MC_ID
			self.h = MC_random(nb_states,self.alphabet,random_initial_state)
		
		elif self.training_set.type == 1:
			from ..mdp.MDP import MDP_random
			self.type_model = MDP_ID
			act,_ = self.training_set.getActionsObservations()
			self.h = MDP_random(nb_states,self.alphabet,act,random_initial_state)
		
		elif self.training_set.type == 3:
			from ..gohmm.GoHMM import GoHMM_random
			if nb_distributions == None:
				nb_distributions = len(self.training_set.sequences[0][0])
			self.type_model = GOHMM_ID
			self.h = GoHMM_random(nb_states,nb_distributions,random_initial_state)
		
		elif self.training_set.type == 4:
			from ..ctmc.CTMC import CTMC_random
			if min_exit_rate_time == None or max_exit_rate_time == None:
				raise RuntimeError("The minimum and maximum exit rate must be given if the initial hypothesis is not provided.")
			self.type_model = CTMC_ID
			self.h = CTMC_random(nb_states,self.alphabet,min_exit_rate_time,max_exit_rate_time,self_loop,random_initial_state)

	def _computeAlphas(self, sequence: list, dtype=False) -> array:
		"""
		Compute the alpha values for ``sequence`` under the current BW
		hypothesis.

		Parameters
		----------
		sequence : list of str
			sequence of observations.
		dtype : numpy.scalar
			If it set, the output will be a numpy array of this type,
			otherwise it is a numpy array of float64.

		Returns
		-------
		2-D narray
			array containing the alpha values.
		"""
		len_seq = len(sequence)
		init_arr = self.h.initial_state
		zero_arr = zeros(shape=(len_seq*self.nb_states,))
		alpha_matrix = append(init_arr,zero_arr).reshape(len_seq+1,self.nb_states)
		if dtype != False:
			alpha_matrix = alpha_matrix.astype(dtype)
		else:
			dtype = float64
		for k in range(len_seq):
			for s in range(self.nb_states):
				p = array([self._h_tau(ss,s,sequence[k]) for ss in range(self.nb_states)],dtype=dtype)
				alpha_matrix[k+1,s] = dot(alpha_matrix[k],p)
		return alpha_matrix.T

	def _computeBetas(self, sequence: list, dtype=False) -> array:
		"""
		Compute the beta values for ``sequence`` under the current BW
		hypothesis.

		Parameters
		----------
		sequence : list of str
			sequence of observations.
		dtype : numpy.scalar
			If it set, the output will be a numpy array of this type,
			otherwise it is a numpy array of float64.

		Returns
		-------
		2-D narray
			array containing the beta values.
		"""
		len_seq = len(sequence)
		init_arr = ones(self.nb_states)
		zero_arr = zeros(shape=(len_seq*self.nb_states,))
		beta_matrix = append(zero_arr,init_arr).reshape(len_seq+1,self.nb_states)
		if dtype != False:
			beta_matrix = beta_matrix.astype(dtype)
		else:
			dtype = float64
		for k in range(len(sequence)-1,-1,-1):
			for s in range(self.nb_states):
				p = array([self._h_tau(s,ss,sequence[k]) for ss in range(self.nb_states)],dtype=dtype)
				beta_matrix[k,s] = dot(beta_matrix[k+1],p)
		return beta_matrix.T

	def _computeTaus(self):
		pass
	
	def _runProcesses(self,training_set):
		if platform != "win32" and platform != "darwin" and NB_PROCESS > 1:
			p = Pool(processes = NB_PROCESS)
			tasks = []
			for seq,times in zip(training_set.sequences,training_set.times):
				tasks.append(p.apply_async(self._processWork, [seq, times,]))
			return [res.get() for res in tasks if res.get() != False]
		else:
			return [self._processWork(seq, times) for seq,times in zip(training_set.sequences,training_set.times)]
	
	def _endPrint(self,it,rt):
		print()
		print("---------------------------------------------")
		print("Learning finished")
		print("Iterations:\t  ",it)
		print("Running time:\t  ",rt)
		print("---------------------------------------------")
		print()

	def _h_tau(self,s1: int,s2: int,obs: str) -> float:
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
		return self.h.tau(s1,s2,obs)
