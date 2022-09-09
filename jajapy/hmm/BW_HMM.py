from .HMM import HMM, HMM_state, HMM_random
from ..base.BW import *
from numpy import log
from ..base.Set import Set

class BW_HMM(BW):
	"""
	class for general Baum-Welch algorithm on HMM.
	This algorithm is described here:
	https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf
	"""
	def __init__(self):
		super().__init__()

	def fit(self, traces: Set, initial_model: HMM=None, nb_states: int=None,
			random_initial_state: bool=False, output_file: str=None,
			epsilon: float=0.01,
			pp: str=''):
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : Set
			training set.
		initial_model : HMM, optional.
			first hypothesis. If not set it will create a random HMM with
			``nb_states`` states. Should be set if ``nb_states`` is not set.
		nb_states: int
			If ``initial_model`` is not set it will create a random HMM with
			``nb_states`` states. Should be set if ``initial_model`` is not set.
			Default is None.
		random_initial_state: bool
			If ``initial_model`` is not set it will create a random HMM with
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
		HMM
			fitted HMM.
		"""
		if not initial_model:
			if not nb_states:
				print("Either nb_states or initial_model should be set")
				return
			initial_model = HMM_random(nb_states,traces.getAlphabet(),random_initial_state)
		self.alphabet = initial_model.getAlphabet()
		return super().fit(traces, initial_model, output_file, epsilon, pp)

	def _processWork(self,sequence,times):
		alpha_matrix = self.computeAlphas(sequence)
		beta_matrix = self.computeBetas(sequence)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq != 0.0:
			####################
			den   = zeros(self.nb_states)
			####################
			num_a = zeros(shape=(self.nb_states,self.nb_states))
			num_b = zeros(shape=(self.nb_states,len(self.alphabet)))
			for s in range(self.nb_states):
				den[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1],times/proba_seq).sum()
				for ss in range(self.nb_states):
					p = array([self.h_tau(s,ss,o) for o in sequence])
					num_a[s,ss] = dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],times/proba_seq).sum()
					for o, obs in enumerate(self.alphabet):
						arr_dirak = [1.0 if t == obs else 0.0 for t in sequence]
						num_b[s,o] = dot(alpha_matrix[s][:-1]*arr_dirak*beta_matrix[s][:-1],times/proba_seq).sum()
			####################
			num_init = alpha_matrix.T[0]*beta_matrix.T[0]*times/proba_seq
			####################
			return [den, num_a, num_b, proba_seq, times, num_init]
		return False

	def _generateHhat(self,temp):
		den = zeros(shape=(self.nb_states,))
		a   = zeros(shape=(self.nb_states,self.nb_states))
		b   = zeros(shape=(self.nb_states,len(self.alphabet)))
		lst_den = array([i[0] for i in temp]).T
		lst_num_a = array([i[1] for i in temp]).T.reshape(self.nb_states*self.nb_states,len(temp))
		lst_num_b = array([i[2] for i in temp]).T.reshape(self.nb_states*len(self.alphabet),len(temp))
		lst_proba=array([i[3] for i in temp])
		lst_times=array([i[4] for i in temp])
		lst_init =array([i[5] for i in temp]).T
		
		currentloglikelihood = dot(log(lst_proba),lst_times)

		for s in range(self.nb_states):
			den[s] = lst_den[s].sum()
			for x in range(self.nb_states):
				a[s,x] = lst_num_a[x*self.nb_states+s].sum()
			for x in range(len(self.alphabet)):
				b[s,x] = lst_num_b[x*self.nb_states+s].sum()

		new_states_t = []
		new_states_o = []
		for s in range(self.nb_states):
			la = a[s]/den[s]
			lb = b[s]/den[s]
			la = list(zip(range(self.nb_states),la))
			lb = list(zip(self.alphabet,lb))
			l = HMM_state(lb, la, self.alphabet, self.nb_states)
			new_states_t.append(l[0])
			new_states_o.append(l[1])
		output = array(new_states_o)
		matrix = array(new_states_t)

		initial_state = [lst_init[s].sum()/lst_init.sum() for s in range(self.nb_states)]
		
		return [HMM(matrix, output, self.alphabet,initial_state),currentloglikelihood]
