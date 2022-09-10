from .CTMC import *
from .BW_CTMC import *


class MM_CTMC_Composition(BW_CTMC):
	def __init__(self) -> None:
		super().__init__()

	def fit(self, traces: Set, initial_model_1: CTMC=None, nb_states_1: int=None,
			random_initial_state_1: bool=False, min_exit_rate_time_1 : int=1.0,
			max_exit_rate_time_1: int=10.0, initial_model_2: CTMC=None,
			nb_states_2: int=None, random_initial_state_2: bool=False,
			min_exit_rate_time_2 : int=1.0,	max_exit_rate_time_2: int=10.0,
			output_file: str=None, epsilon: float=0.01, pp: str='',
			to_update: int=None):
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : Set
			training set.
		initial_model_1 : CTMC, optional.
			First hypothesis. If not set it will create a random CTMC with
			`nb_states_1` states. Should be set if `nb_states_1` is not set.
		nb_states_1: int, optional
			If ``initial_model_1`` is not set it will create a random CTMC with
			`nb_states_1` states. Should be set if `initial_model_1` is not set.
		random_initial_state_1: bool
			If ``initial_model_1`` is not set it will create a random CTMC with
			random initial state according to this sequence of probabilities.
			Default is False.
		min_exit_rate_time_1: int, optional
			Minimum exit rate for the states in the first hypothesis if
			``initial_model_1`` is not set.
			Default is 1.0.
		max_exit_rate_time_1: int, optional
			Maximum exit rate for the states in the first hypothesis if
			``initial_model_1`` is not set.
			Default is 10.0.
		initial_model_2 : CTMC, optional.
			Second hypothesis. If not set it will create a random CTMC with
			`nb_states_2` states. Should be set if `nb_states_2` is not set.
		nb_states_2: int, optional
			If ``initial_model_2`` is not set it will create a random CTMC with
			`nb_states_2` states. Should be set if `initial_model_2` is not set.
		random_initial_state_2: bool
			If ``initial_model_2`` is not set it will create a random CTMC with
			random initial state according to this sequence of probabilities.
			Default is False.
		min_exit_rate_time_2: int, optional
			Minimum exit rate for the states in the second hypothesis if
			``initial_model_2`` is not set.
			Default is 1.0.
		max_exit_rate_time_2: int, optional
			Maximum exit rate for the states in the second hypothesis if
			``initial_model_2`` is not set.
			Default is 10.0.
		output_file : str, optional
			if set path file of the output model. Otherwise the output model
			will not be saved into a text file.
		epsilon : float, optional
			the learning process stops when the difference between the
			loglikelihood of the training set under the two last hypothesis is
			lower than ``epsilon``. The lower this value the better the output,
			but the longer the running time. By default 0.01.
		pp : str, optional
			Will be printed at each iteration. By default ''.
		to_update: int, optional
			If set to 1 only the first hypothesis will be updated,
			If set to 2 only the second hypothesis will be updated,
			If set to None the two hypothesis will be updated.
			Default is None.

		Returns
		-------
		CTMC
			fitted CTMC.
		"""
		if not initial_model_1:
			if not nb_states_1:
				print("Either nb_states_1 or initial_model_1 should be set")
				return
			initial_model_1 = CTMC_random(nb_states_1,
										traces.getAlphabet(),
										min_exit_rate_time_1, max_exit_rate_time_1,
										False, random_initial_state_1)
		if not initial_model_2:
			if not nb_states_2:
				print("Either nb_states_2 or initial_model_2 should be set")
				return
			initial_model_2 = CTMC_random(nb_states_2,
										traces.getAlphabet(),
										min_exit_rate_time_2, max_exit_rate_time_2,
										False, random_initial_state_2)

		initial_model = asynchronousComposition(initial_model_1,initial_model_2)
		self.hs = [None,initial_model_1,initial_model_2]
		self.nb_states_hs = [None,initial_model_1.nb_states,initial_model_2.nb_states]
		self.alphabets = [None,self.hs[1].getAlphabet(),self.hs[2].getAlphabet()]
		self.disjoints_alphabet = len(set(self.alphabets[1]).intersection(set(self.alphabets[2]))) == 0
		self.to_update = to_update
		super().fit(traces, initial_model, output_file, epsilon, pp)
		if output_file:
			self.hs[1].save(output_file+"_1.txt")
			self.hs[2].save(output_file+"_2.txt")
		return self.hs[1], self.hs[2]

	def _getStateInComposition(self,s:int,model:int,s2:int=0):
		if model == 1:
			return s*self.nb_states_hs[2]+s2
		else:
			return s2*self.nb_states_hs[2]+s

	def _oneSequence(self,obs_seq,times_seq,times,timed,alpha_matrix,beta_matrix,to_update,proba_seq) -> list:
		other = to_update%2 + 1
		nb_states = self.nb_states_hs[to_update]
		nb_states_other = self.nb_states_hs[other]
		den = zeros(nb_states)
		num = zeros(shape=(nb_states,nb_states*len(self.alphabet)))
		for v in range(nb_states):
			for u in range(nb_states_other):
				uv = self._getStateInComposition(v,to_update,u)
				if self.disjoints_alphabet:
					divider = self.hs[to_update].e(v)
				else:
					divider = self.hs[to_update].e(v) + self.hs[other].e(u)
				if timed:
					den[v] += dot(alpha_matrix[uv][:-1]*beta_matrix[uv][:-1]*times_seq,times/proba_seq).sum()
				else:
					den[v] += dot(alpha_matrix[uv][:-1]*beta_matrix[uv][:-1],times/proba_seq).sum()
				
				for vv in [i for i in range(nb_states) if i != v]:
					uvv = self._getStateInComposition(vv,to_update,u)
					if timed:
						p = array([exp(-divider*t)*self.hs[to_update].l(v,vv,o) for o,t in zip(obs_seq,times_seq)])
					else:
						p = array([self.hs[to_update].l(v,vv,o) for o in obs_seq])
					c = 0
					for obs in self.alphabets[to_update]:
						arr_dirak = [1.0 if o == obs else 0.0 for o in obs_seq]
						num[v,vv*len(self.alphabets[to_update])+c] += dot(alpha_matrix[uv][:-1]*arr_dirak*beta_matrix[uvv][1:]*p,times/proba_seq).sum()
						c += 1
		num_init = alpha_matrix.T[0]*beta_matrix.T[0]*times/proba_seq
		return [den,num,num_init]

	def _computeAlphasBetas(self,obs_seq: list, times_seq: list=None) -> tuple:
		"""
		Computes the alpha and the beta matrix for a given a trace (timed or
		not).

		Parameters
		----------
		obs_seq : list of str
			Sequence of observations.
		times_seq : list of float, optional
			Sequence of waiting times (omitted is the sequence is non-timed).

		Returns
		-------
		tuple
			The alpha matrix and the beta matrix.
		"""
		if not self.disjoints_alphabet:
			return self.computeAlphas(obs_seq, times_seq), self.computeBetas(obs_seq, times_seq)
		else:
			obs_seqs, times_seq = self._splitSequenceObs(obs_seq, times_seq)
			bw = BW_CTMC(self.hs[1])
			alphas1 = bw.computeAlphas(obs_seqs[1], times_seq[0])
			betas1  = bw.computeBetas( obs_seqs[1], times_seq[0])
			bw = BW_CTMC(self.hs[2])
			alphas2 = bw.computeAlphas(obs_seqs[2], times_seq[1])
			betas2  = bw.computeBetas( obs_seqs[2], times_seq[1])
			alpha_matrix = zeros(shape=(self.nb_states_hs[1]*self.nb_states_hs[1],len(obs_seq+1)))
			beta_matrix  = zeros(shape=(self.nb_states_hs[1]*self.nb_states_hs[1],len(obs_seq+1)))
			for s1 in range(self.nb_states_hs[1]):
				for s2 in range(self.nb_states_hs[2]):
					alpha_matrix[self._getStateInComposition(s1,1,s2)] = array([alphas1[s1][obs_seqs[0][t]]*alphas2[s2][t-obs_seqs[0][t]] for t in range(len(obs_seq)+1)])
					beta_matrix[self._getStateInComposition(s1,1,s2)] = array( [ betas1[s1][obs_seqs[0][t]]* betas2[s2][t-obs_seqs[0][t]] for t in range(len(obs_seq)+1)])
			return alpha_matrix, beta_matrix


	def _processWork(self,sequence: list, times: int):
		times_seq, obs_seq = self.splitTime(sequence)
		if times_seq == None:
			timed = False
		else:
			timed = True
		alpha_matrix, beta_matrix = self._computeAlphasBetas(obs_seq,times_seq)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq <= 0.0:
			return False

		if self.to_update:
			res1 = self._oneSequence(obs_seq,times_seq,times,timed,alpha_matrix,beta_matrix,self.to_update,proba_seq)
			res2 = None
		else:
			res1 = self._oneSequence(obs_seq,times_seq,times,timed,alpha_matrix,beta_matrix,1,proba_seq)
			res2 = self._oneSequence(obs_seq,times_seq,times,timed,alpha_matrix,beta_matrix,2,proba_seq)
		
		return [res1, res2, proba_seq, times]

	def _generateModel(self,temp,to_update):
		#temp = [[den,num,num_init],[den,num,num_init],...]
		nb_states = self.nb_states_hs[to_update]
		alphabet = self.alphabets[to_update]

		den = array([i[0] for i in temp]).sum(axis=0)
		num = array([i[1] for i in temp]).sum(axis=0)
		lst_init =array([i[2] for i in temp]).T

		list_sta = []
		for i in range(nb_states):
			for _ in alphabet:
				list_sta.append(i)
		list_obs = alphabet*nb_states
		new_states = []
		for s in range(nb_states):
			if den[s] != 0.0:
				l = list(zip(list_sta, list_obs, (num[s]/den[s]).tolist()))
				l = _removeZeros(l)		
				new_states.append(CTMC_state(l,
											self.hs[to_update].alphabet,
											self.hs[to_update].nb_states))
			else:
				new_states.append(self.hs[to_update].matrix[s])

		initial_state = [lst_init[s].sum()/lst_init.sum() for s in range(nb_states)]
		return CTMC(array(new_states),self.hs[to_update].alphabet,initial_state)

	def _generateHhat(self,temp) -> list:
		if self.to_update == 1:
			self.hs[1] = self._generateModel([i[0] for i in temp],1)
		elif self.to_update == 2:
			self.hs[2] = self._generateModel([i[0] for i in temp],2)
		else:
			self.hs[1] = self._generateModel([i[0] for i in temp],1)
			self.hs[2] = self._generateModel([i[1] for i in temp],2)

		currentloglikelihood = sum([log(i[2])*i[3] for i in temp])

		return [asynchronousComposition(self.hs[1],self.hs[2]),currentloglikelihood]

	def _splitSequenceObs(self,seq,tseq):
		res0 = [0] #nb obs for model 1 seen until now
				   #[0,0,0,...0] only obs for model 2
				   #[0,1,2,...n] only obs for model 1
		res1 = []
		res2 = []
		t1   = []
		t2   = []
		for i,o in enumerate(seq):
			res0.append(res0[-1])
			if o in self.alphabets[1]:
				res1.append(o)
				res0[-1] += 1
				if tseq:
					t1.append(tseq[i])
			elif o in self.alphabets[2]:
				res2.append(o)
				if tseq:
					t2.append(tseq[i])
			else:
				input("ERR0R: "+o+" is not in any alphabet")
		return ((res0,res1,res2),(t1,t2))

def _removeZeros(l):
	for i in l:
		if i[2] == 0.0:
			l.remove(i)
	return l