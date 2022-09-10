from .MGOHMM import *
from ..gohmm.BW_GOHMM import *
from numpy import log, sqrt


class BW_MGOHMM(BW_GOHMM):
	"""
	class for general Baum-Welch algorithm on MGOHMM.
	"""
	def __init__(self):
		super().__init__()

	def fit(self, traces: Set, initial_model: MGOHMM=None, nb_states: int=None,
			nb_distributions: int=None, random_initial_state: bool=False,
			output_file: str=None, epsilon: float=0.01, pp: str=''):
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : Set
			training set.
		initial_model : MGOHMM, optional.
			first hypothesis. If not set it will create a random MGOHMM with
			``nb_states`` states. Should be set if ``nb_states`` is not set.
		nb_states: int, optional.
			If ``initial_model`` is not set it will create a random MGOHMM with
			``nb_states`` states. Should be set if ``initial_model`` is not set.
			Default is None.
		nb_distributions: int, optional.
			If ``initial_model`` is not set it will create a random MGOHMM with
			``nb_distributions`` distributions in each state.
			Should be set if ``initial_model`` is not set.
		random_initial_state: bool
			If ``initial_model`` is not set it will create a random MGOHMM with
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
		pp : str, optional
			Will be printed at each iteration. By default ''

		Returns
		-------
		MGOHMM
			fitted MGOHMM.
		"""
		if not initial_model:
			if not nb_states:
				print("Either nb_states or initial_model should be set")
				return
			if not nb_distributions:
				print("Either nb_distributions or initial_model should be set")
				return
			initial_model = MGOHMM_random(nb_states,nb_distributions,
										  random_initial_state)
		self.nb_distr = initial_model.nb_distributions
		return super().fit(traces, initial_model, output_file, epsilon, pp)

	def _processWork(self,sequence,times):
		sequence = array(sequence)
		alpha_matrix = self.computeAlphas(sequence)
		beta_matrix = self.computeBetas(sequence)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq != 0.0:
			den    = zeros(self.nb_states)
			num_a  = zeros(shape=(self.nb_states,self.nb_states))
			num_mu = zeros(shape=(self.nb_states,self.nb_distr))
			num_va = zeros(shape=(self.nb_states,self.nb_distr))
			for s in range(self.nb_states):
				den[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1],times/proba_seq).sum()
				num_mu[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*sequence.T,times/proba_seq).sum(axis=1)
				num_va[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*(sequence-self.h.mu(s)).T**2,times/proba_seq).sum(axis=1)
				for ss in range(self.nb_states):
					p = array([self.h_tau(s,ss,o) for o in sequence])
					num_a[s,ss] = dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],times/proba_seq).sum()
			num_init = alpha_matrix.T[0]*beta_matrix.T[0]*times/proba_seq
			return [den,num_a,num_mu,num_va,proba_seq,times,num_init]
		return False

	def _generateHhat(self,temp):
		a  = array([i[1] for i in temp]).sum(axis=0)
		den= array([i[0] for i in temp]).sum(axis=0)
		mu = array([i[2] for i in temp]).sum(axis=0)
		va = array([i[3] for i in temp]).sum(axis=0)
		lst_proba=array([i[4] for i in temp])
		lst_times=array([i[5] for i in temp])
		init =array([i[6] for i in temp]).sum(axis=0)


		currentloglikelihood = dot(log(lst_proba),lst_times)

		mu = (mu.T/den).T
		va = sqrt((va.T/den).T)
		a  = (a.T /den).T

		matrix = []
		output = []
		for s in range(self.nb_states):
			la = list(zip(list(range(self.nb_states)),a[s].tolist()))
			l = MGOHMM_state(la,array([mu[s],va[s]]).T.tolist(),self.nb_states)
			matrix.append(l[0])
			output.append(l[1])

		initial_state = [init[s]/init.sum() for s in range(self.nb_states)]
		
		return [MGOHMM(array(matrix),array(output),initial_state),currentloglikelihood]