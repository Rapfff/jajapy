from .GOHMM import *
from ..base.BW import *
from ..base.Set import Set
from numpy import log


class BW_GOHMM(BW):
	"""
	class for general Baum-Welch algorithm on GOHMM.
	This algorithm is described here:
	http://www.inass.org/2020/2020022920.pdf
	"""
	def __init__(self):
		super().__init__()

	def fit(self, traces: Set, initial_model: GOHMM=None, nb_states: int=None,
			random_initial_state: bool=False, output_file: str=None,
			epsilon: float=0.01,
			pp: str=''):
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : list
			training set.
		initial_model : GOHMM, optional.
			first hypothesis. If not set it will create a random GOHMM with
			``nb_states`` states. Should be set if ``nb_states`` is not set.
		nb_states: int
			If ``initial_model`` is not set it will create a random GOHMM with
			``nb_states`` states. Should be set if ``initial_model`` is not set.
			Default is None.
		random_initial_state: bool
			If ``initial_model`` is not set it will create a random GOHMM with
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
		GOHMM
			fitted GOHMM.
		"""
		if not initial_model:
			if not nb_states:
				print("Either nb_states or initial_model should be set")
				return
			initial_model = GOHMM_random(nb_states,random_initial_state)
		return super().fit(traces, initial_model, output_file, epsilon, pp)

	def _processWork(self,sequence,times):
		sequence = array(sequence)
		alpha_matrix = self.computeAlphas(sequence)
		beta_matrix = self.computeBetas(sequence)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq != 0.0:
			den    = zeros(self.nb_states)
			num_a  = zeros(shape=(self.nb_states,self.nb_states))
			num_mu = zeros(self.nb_states)
			num_va = zeros(self.nb_states)
			for s in range(self.nb_states):
				den[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1],times/proba_seq).sum()
				num_mu[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*sequence,times/proba_seq).sum()
				num_va[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*(sequence-self.h.mu(s))**2,times/proba_seq).sum()
				for ss in range(self.nb_states):
					p = array([self.h_tau(s,ss,o) for o in sequence])
					num_a[s,ss] = dot(alpha_matrix[s][:-1]*p*beta_matrix[ss][1:],times/proba_seq).sum()
			num_init = alpha_matrix.T[0]*beta_matrix.T[0]*times/proba_seq
			return [den,num_a,num_mu,num_va,proba_seq,times,num_init]
		print('proba0')
		return False

	def _generateHhat(self,temp):
		a   = zeros(shape=(self.nb_states,self.nb_states))
		den = array([i[0] for i in temp]).sum(axis=0)
		lst_num_a = array([i[1] for i in temp]).T.reshape(self.nb_states*self.nb_states,len(temp))
		mu = array([i[2] for i in temp]).sum(axis=0)
		va = array([i[3] for i in temp]).sum(axis=0)
		lst_proba=array([i[4] for i in temp])
		lst_times=array([i[5] for i in temp])
		init =array([i[6] for i in temp]).sum(axis=0)

		currentloglikelihood = dot(log(lst_proba),lst_times)

		matrix = []
		output = []
		for s in range(self.nb_states):
			for x in range(self.nb_states):
				a[s,x] = lst_num_a[x*self.nb_states+s].sum()
			la = list(zip(list(range(self.nb_states)) ,a[s]/den[s]))
			l = GOHMM_state(la,(mu[s]/den[s],sqrt(va[s]/den[s])),self.nb_states)
			matrix.append(l[0])
			output.append(l[1])

		initial_state = [init[s]/init.sum() for s in range(self.nb_states)]
		
		return [GOHMM(array(matrix),array(output),initial_state),currentloglikelihood]