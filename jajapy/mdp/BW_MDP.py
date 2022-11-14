from .MDP import *
from ..base.BW import *
from ..base.Set import Set
from ..base.tools import normalize
from numpy import array, dot, append, zeros, ones, log, inf

NB_PROCESS = 11

class BW_MDP(BW):
	"""
	Class for general Passive Baum-Welch algorithm on MDP.
	This algorithm is described here:
	https://arxiv.org/pdf/2110.03014.pdf
	"""
	
	def __init__(self):
		super().__init__()
	
	def fit(self, traces, initial_model=None, nb_states: int=None,
			random_initial_state: bool=False, output_file: str=None,
			epsilon: float=0.01, max_it: int= inf, pp: str='',
			verbose: bool = True, return_data: bool= False, stormpy_output: bool = True):
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : Set or list or numpy.ndarray
			training set.
		initial_model : MDP or stormpy.SparseCtmc, optional.
			first hypothesis. If not set it will create a random MDP with
			``nb_states`` states. Should be set if ``nb_states`` is not set.
		nb_states: int
			If ``initial_model`` is not set it will create a random MDP with
			``nb_states`` states. Should be set if ``initial_model`` is not set.
			Default is None.
		random_initial_state: bool
			If ``initial_model`` is not set it will create a random MDP with
			random initial state according to this sequence of probabilities.
			Should be set if ``initial_model`` is not set.
			Default is False.
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
			Will be printed at each iteration. By default ''
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
		MDP or stormpy.SparseMdp
			The fitted MDP.
			If `stormpy_output` is set to `False` or if stormpy is not available on
			the machine it returns a `jajapy.MDP`, otherwise it returns a `stormpy.SparseMdp`
		"""
		if type(traces) != Set:
			traces = Set(traces, t=1)
		if not initial_model:
			if not nb_states:
				print("Either nb_states or initial_model should be set")
				return
			actions, observations = traces.getActionsObservations()
			initial_model = MDP_random(nb_states,observations,actions,random_initial_state)
		else:
			observations = initial_model.getAlphabet()
			actions = initial_model.getActions()
		self.alphabet = observations
		self.actions = actions
		return super().fit(traces, initial_model, output_file, epsilon, max_it, pp, verbose,return_data,stormpy_output)

	def h_tau(self,s1: int,act: str,s2: int,obs: str) -> float:
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

	def computeAlphas(self,sequence: list, sequence_actions: list) -> array:
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
				p = array([self.h_tau(ss,sequence_actions[k],s,sequence[k]) for ss in range(self.nb_states)])
				alpha_matrix[k+1,s] = dot(alpha_matrix[k],p)
		
		alpha_matrix[-1] *= (array(self.h.labeling) == sequence[-1])
		return alpha_matrix.T

	def computeBetas(self,sequence: list,sequence_actions: list) -> array:
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
		init_arr = ones(self.nb_states)*(array(self.h.labeling) == sequence[-1])
		zero_arr = zeros(shape=((len_seq-1)*self.nb_states,))
		beta_matrix = append(zero_arr,init_arr).reshape(len_seq,self.nb_states)
		for k in range(len(sequence)-2,-1,-1):
			for s in range(self.nb_states):
				p = array([self.h_tau(s,sequence_actions[k],ss,sequence[k]) for ss in range(self.nb_states)])
				beta_matrix[k,s] = dot(beta_matrix[k+1],p)
		return beta_matrix.T

	def _processWork(self,sequence,times):
		sequence_actions = [sequence[i+1] for i in range(0,len(sequence)-1,2)]
		sequence_obs = [sequence[i] for i in range(0,len(sequence)-1,2)]+[sequence[-1]]
		alpha_matrix = self.computeAlphas(sequence_obs,sequence_actions)
		beta_matrix = self.computeBetas(sequence_obs,sequence_actions)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq != 0.0:
			den = zeros(shape=(self.nb_states,len(self.actions)))
			num = zeros(shape=(self.nb_states,len(self.actions),self.nb_states))
			
			for s in range(self.nb_states):
				for i,a in enumerate(self.actions):
					arr_dirak = array([1.0 if t == a else 0.0 for t in sequence_actions])
					den[s,i] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*arr_dirak,times/proba_seq).sum()
			
				for ss in range(self.nb_states):
					p = array([self.h_tau(s,a,ss,o) for o,a in zip(sequence_obs[:-1],sequence_actions)])
					for ia,act in enumerate(self.actions):
						arr_dirak = [1.0 if a == act else 0.0 for a in sequence_actions]
						num[s,ia,ss] = dot(alpha_matrix[s][:-1]*arr_dirak*beta_matrix[ss][1:]*p,times/proba_seq).sum()
			
			num_init = alpha_matrix.T[0]*beta_matrix.T[0]*times/proba_seq
			return [den,num,proba_seq,times,num_init]
		return False

	def _generateHhat(self,temp):
		den = array([i[0] for i in temp]).sum(axis=0)
		num = array([i[1] for i in temp]).sum(axis=0)
		lst_proba=array([i[2] for i in temp])
		lst_times=array([i[3] for i in temp])
		lst_init =array([i[4] for i in temp]).T

		currentloglikelihood = dot(log(lst_proba),lst_times)

		for s,a in zip(range(self.nb_states),range(len(self.actions))):
			if den[s][a] == 0.0:
				den[s][a] = 1.0
				num[s][a] = self.h.matrix[s][a]

		den = den.reshape(self.nb_states,len(self.actions),1)
		matrix = num/den

		initial_state = normalize([lst_init[s].sum()/lst_init.sum() for s in range(self.nb_states)])

		return [MDP(matrix,self.h.labeling,self.h.actions,initial_state), currentloglikelihood]
		