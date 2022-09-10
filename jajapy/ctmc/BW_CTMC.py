from .CTMC import *
from ..base.BW import BW
from ..base.Set import Set
from numpy import array, zeros, dot, append, ones, log


class BW_CTMC(BW):
	def __init__(self) -> None:
		super().__init__()

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
		len_seq = len(obs_seq)
		init_arr = self.h.initial_state
		zero_arr = zeros(shape=(len_seq*self.nb_states,))
		alpha_matrix = append(init_arr,zero_arr).reshape(len_seq+1,self.nb_states)
		for k in range(len_seq):
			for s in range(self.nb_states):
				p = array([self.h_l(ss,s,obs_seq[k])*exp(-self.h_e(ss)*times_seq[k]) for ss in range(self.nb_states)])
				alpha_matrix[k+1,s] = dot(alpha_matrix[k],p)
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
		len_seq = len(obs_seq)
		init_arr = ones(self.nb_states)
		zero_arr = zeros(shape=(len_seq*self.nb_states,))
		beta_matrix = append(zero_arr,init_arr).reshape(len_seq+1,self.nb_states)
		for k in range(len(obs_seq)-1,-1,-1):
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


	def fit(self, traces: Set, initial_model: CTMC=None, nb_states: int=None,
			random_initial_state: bool=False, min_exit_rate_time : int=1.0,
			max_exit_rate_time: int=10.0, self_loop: bool = True,
			output_file: str=None, epsilon: float=0.01, pp: str=''):
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : Set
			training set.
		initial_model : CTMC, optional.
			first hypothesis. If not set it will create a random CTMC with
			``nb_states`` states. Should be set if ``nb_states`` is not set.
		nb_states: int
			If ``initial_model`` is not set it will create a random CTMC with
			``nb_states`` states. Should be set if ``initial_model`` is not set.
			Default is None.
		random_initial_state: bool
			If ``initial_model`` is not set it will create a random CTMC with
			random initial state according to this sequence of probabilities.
			Should be set if ``initial_model`` is not set.
			Default is False.
		min_exit_rate_time: int, optional
			Minimum exit rate for the states in the first hypothesis.
			Default is 1.0.
		max_exit_rate_time: int, optional
			Minimum exit rate for the states in the first hypothesis.
			Default is 10.0.
		self_loop: bool, optional
			Wether or not there will be self loop in the first hypothesis.
			Default is True.
		output_file : str, optional
			if set path file of the output model. Otherwise the output model
			will not be saved into a text file.
		epsilon : float, optional
			the learning process stops when the difference between the
			loglikelihood of the training set under the two last hypothesis is
			lower than ``epsilon``. The lower this value the better the output,
			but the longer the running time. By default 0.01.
		pp : str, optional
			Will be printed at each iteration. By default ''

		Returns
		-------
		CTMC
			fitted CTMC.
		"""
		if not initial_model:
			if not nb_states:
				print("Either nb_states or initial_model should be set")
				return
			initial_model = CTMC_random(nb_states,
										traces.getAlphabet(),
										min_exit_rate_time, max_exit_rate_time,
										self_loop, random_initial_state)
		self.alphabet = initial_model.getAlphabet()
		return super().fit(traces, initial_model, output_file, epsilon, pp)


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
		if type(sequence[0]) == float and type(sequence[1]) == str:
			times_seq = [sequence[i] for i in range(0,len(sequence),2)]
			obs_seq   = [sequence[i] for i in range(1,len(sequence),2)]
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
		for s in range(self.nb_states):
			den = zeros(self.nb_states)
			num = zeros(shape=(self.nb_states,self.nb_states*len(self.alphabet)))

			if timed:
				den[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*times_seq,times/proba_seq).sum()
			else:
				den[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1],times/proba_seq).sum()
			c = 0
			for ss in range(self.nb_states):
				if timed:
					p = array([self.h_l(s,ss,o)*exp(-self.h_e(s)*t) for o,t in zip(obs_seq,times_seq)])
				else:
					p = array([self.h_l(s,ss,o) for o in obs_seq])
				for obs in self.alphabet:
					arr_dirak = [1.0 if o == obs else 0.0 for o in obs_seq]
					num[s,c] = dot(alpha_matrix[s][:-1]*arr_dirak*beta_matrix[ss][1:]*p,times/proba_seq).sum()
					c += 1
		####################			
		num_init = alpha_matrix.T[0]*beta_matrix.T[0]*times/proba_seq
		####################
		return [den, num, proba_seq, times, num_init]

	def _generateHhat(self,temp):
		den = array([i[0] for i in temp]).sum(axis=0)
		num = array([i[1] for i in temp]).sum(axis=0)
		lst_proba=array([i[2] for i in temp])
		lst_times=array([i[3] for i in temp])
		lst_init =array([i[4] for i in temp]).T

		currentloglikelihood = dot(log(lst_proba),lst_times)

		list_sta = []
		for i in range(self.nb_states):
			for _ in self.alphabet:
				list_sta.append(i)
		list_obs = self.alphabet*self.nb_states
		new_states = []

		for s in range(self.nb_states):
			if den[s] != 0.0:
				l = list(zip(list_sta, list_obs, num[s]/den[s]))
				new_states.append(CTMC_state(l,self.alphabet,self.nb_states))
			else:
				new_states.append(self.h.matrix[s])
		initial_state = [lst_init[s].sum()/lst_init.sum() for s in range(self.nb_states)]
		return [CTMC(array(new_states),self.alphabet,initial_state),currentloglikelihood]
