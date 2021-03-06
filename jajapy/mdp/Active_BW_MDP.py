from .Scheduler import UniformScheduler, MemorylessScheduler
from .BW_MDP import BW_MDP
from ..base.BW import NB_PROCESS
from ..base.Set import Set
from .MDP import MDP
from ..base.tools import resolveRandom
from multiprocessing import Pool
from random import random
from numpy import zeros, dot, array, argmin
from sys import platform
class ActiveLearningScheduler:
	"""
	Class for a scheduler used during the active learning sampling.
	"""
	def __init__(self,memoryless_scheduler: MemorylessScheduler,m: MDP) -> None:
		"""
		Create an ActiveLearningScheduler.

		Parameters
		----------
		memoryless_scheduler : MemorylessScheduler
			A memoryless scheduler generated by Active_BW_MDP._strategy
		m : MDP
			The current hypothesis.
		"""
		self.m = m
		self.nb_states = len(self.m.states)
		self.memoryless_scheduler = memoryless_scheduler
		self.reset()
		
	def reset(self) -> None:
		"""
		Resets the scheduler. Should be done before genereting a new trace.
		"""
		self.last_obs = None
		self.last_act = None
		self.alpha_matrix = array([self.m.pi(s) for s in range(self.nb_states)])

	def getAction(self) -> str:
		"""
		Returns the action chosen by the scheduler

		Returns
		-------
		str
			Action chosen by the scheduler.
		"""
		if self.last_obs != None:
			alpha_matrix = zeros(self.nb_states)
			for s in range(self.nb_states):
				p = array([self.m.tau(ss,self.last_act,s,self.last_obs) for ss in range(self.nb_states)])
				alpha_matrix[s] = dot(self.alpha_matrix,p)
			self.alpha_matrix = alpha_matrix

		tot = self.alpha_matrix.sum()
		if tot <= 0.0:
			t = [1/len(t) for i in self.alpha_matrix]
		else:
			t = [i/tot for i in self.alpha_matrix]
		s_i = resolveRandom(t)
		act = self.memoryless_scheduler.getAction(s_i)
		self.last_act = act
		return act

	def addObservation(self,obs: str) -> None:
		"""
		Updates the scheduler according to the given observation ``obs``.

		Parameters
		----------
		obs : str
			An observation.
		"""
		self.last_obs = obs

class Active_BW_MDP(BW_MDP):
	"""
	Class for general Active Baum-Welch algorithm on MDP.
	This algorithm is described here:
	https://arxiv.org/pdf/2110.03014.pdf
	"""
	def __init__(self):
		super().__init__()

	def fit(self,traces: Set, lr, nb_iterations: int, nb_sequences: int,
			sequence_length: int = None, epsilon_greedy: float = 0.9,
			initial_model: MDP=None, nb_states: int=None,
			random_initial_state: bool=False, output_file: str=None,
			epsilon: float=0.01, pp: str='') -> MDP:
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : Set
			Training set.
		lr : float, or ``0``, or ``"dynamic"``
			Learning rate. If ``lr=0`` the current hypothesis will be updated
			on ``traces`` plus all the active samples. If `lr` is a float, the
			current hypothesis will be merged with an updated version of it.
			The updated version is a version of the current hypothesis fitted
			on the last active sample. If `lr="dynamic"` the learning rate
			decrease at each iteration.
		nb_iterations: int
			Number of active learning iterations.
		nb_sequences: int
			Size of one active sample. I.e: number of sequences generated at
			each active learning iteration.
		sequence_length: int, optional.
			Size of one sequence generated during an active learning iteration.
			If not set it will be equal to the size of the first sequence in
			``traces``.
		epsilon_greedy: float, optional.
			It will use the active learning scheduler with probability
			``epsilon_greedy`` while genereting the active learning sample, and
			with probability `1-epsilon_greedy` it will use an uniform scheduler.
			By default 0.9
		initial_model : MDP, optional.
			First hypothesis. If not set it will create a random MDP with
			``nb_states`` states. Should be set if ``nb_states`` is not set.
		nb_states: int
			If ``initial_model`` is not set it will create a random MDP with
			``nb_states`` states. Should be set if ``initial_model`` is not set.
			Default is None.
		random_initial_state: bool, optional.
			If ``initial_model`` is not set it will create a random MDP with
			random initial state according to this sequence of probabilities.
			Should be set if ``initial_model`` is not set.
			Default is False.
		output_file : str, optional
			If set path file of the output model. Otherwise the output model
			will not be saved into a text file.
		epsilon : float, optional
			The learning process stops when the difference between the
			loglikelihood of the training set under the two last hypothesis is
			lower than ``epsilon``. The lower this value the better the output,
			but the longer the running time. By default 0.01.
		pp : str, optional
			Will be printed at each iteration. By default ''

		Returns
		-------
		MDP
			fitted MDP.
		"""

		super().fit(traces, initial_model,nb_states,
					random_initial_state, output_file,epsilon,"Passive iteration:"+pp)
	
		total_traces = traces
		
		if sequence_length == None:
			sequence_length = int(len(traces.sequences[0])/2)

		c = 1
		while c <= nb_iterations :
			traces = self._addTraces(sequence_length,nb_sequences,total_traces,epsilon_greedy)
			total_traces.addSet(traces)
			
			if lr == 0:
				super().fit(total_traces, initial_model=self.h,
							output_file=output_file, epsilon=epsilon,
							pp="Active iteration "+str(c)+"/"+str(nb_iterations)+": "+pp)
			else:
				if lr == "dynamic":
					lr_it = sum(traces.times)/(sum(total_traces.times)-sum(traces.times))
				else:
					lr_it = lr
				old_h = self.h
				super().fit(traces, initial_model=self.h,
							output_file=output_file, epsilon=epsilon,
							pp="Active iteration "+str(c)+"/"+str(nb_iterations)+": "+pp)
				self._mergeModels(old_h,lr_it)

			c += 1
		
		return self.h

	def _mergeModels(self,old_h: MDP,lr: float) -> None:
		"""
		Merges the model ``self.h`` with model ``old_h`` using the learning
		rate ``lr``.

		Parameters
		----------
		old_h : MDP
			A MDP to merge with ``self.h``. 
		lr : float
			Learning rate. It is the weight of ``self.h`` in the output model.
		"""
		new_h = self.h
		for s in range(self.nb_states):
			for a in new_h.states[s].actions():
				for p in range(len(new_h.states[s].transition_matrix[a][0])):
					o = new_h.states[s].transition_matrix[a][2][p]
					sprime = new_h.states[s].transition_matrix[a][1][p]
					self.h.states[s].transition_matrix[a][0][p] = (1-lr)*old_h.tau(s,a,sprime,o)+lr*new_h.tau(s,a,sprime,o)

			for a in [i for i in old_h.states[s].actions() if not i in new_h.states[s].actions()]:
				self.h.states[s].transition_matrix[a] = old_h.states[s].transition_matrix[a]

	def _addTraces(self,sequence_length: int,nb_sequence: int,
				  traces: Set,epsilon_greedy: float) -> list:
		"""
		Generates an active learning sample.

		Parameters
		----------
		sequence_length : int
			Length of each sequence in the sample.
		nb_sequence : int
			Number of sequences in the sample.
		traces : Set
			Training set until now. Used to compute the active learning
			sampling strategy.
		epsilon_greedy : It will use the active learning scheduler with probability
			``epsilon_greedy`` and use an uniform scheduler with probability
			`1-epsilon_greedy`.
			
		Returns
		-------
		list
			The active learning sample
		"""
		memoryless_scheduler = self._strategy(traces)
		scheduler_exploit = ActiveLearningScheduler(memoryless_scheduler,self.h)
		scheduler_explore = UniformScheduler(self.h.actions())

		traces = Set([],t=1)
		for n in range(nb_sequence):			
			if random() > epsilon_greedy:
				seq = self.h.run(sequence_length,scheduler_explore)
			else:
				seq = self.h.run(sequence_length,scheduler_exploit)

			if not seq in traces.sequences:
				traces.sequences.append(seq)
				traces.times.append(0)
			traces.times[traces.sequences.index(seq)] += 1
		return traces

	def _strategy(self,traces:Set) -> MemorylessScheduler:
		if platform != "win32":
			p = Pool(processes = NB_PROCESS)
			tasks = []
			for seq,times in zip(traces.sequences,traces.times):
				tasks.append(p.apply_async(self._computeProbas, [seq, times,]))
			temp = array([res.get() for res in tasks])
		else:
			temp = array([self._computeProbas(seq, times) for seq,times in zip(traces.sequences,traces.times)])
		temp = temp.sum(axis=0)
		scheduler = [self.h.actions()[argmin(temp[s])] for s in range(self.nb_states)]
		return MemorylessScheduler(scheduler)

	def _computeProbas(self,seq:list,time:int) -> array:
		sequence_actions = [seq[i] for i in range(0,len(seq),2)]
		sequence_obs = [seq[i+1] for i in range(0,len(seq),2)]
		alpha_matrix = self.computeAlphas(sequence_obs,sequence_actions).T[:-1].T
		res = zeros(shape=(len(self.actions),self.nb_states))
		fact = alpha_matrix.sum(axis=0) # fact[t] proba to generate sequence_obs[:t]

		for ia,a in enumerate(self.actions):
			arr_dirak = array([1.0 if t == a else 0.0 for t in sequence_actions])
			arr_dirak = arr_dirak*time/fact
			res[ia] = dot(alpha_matrix,arr_dirak)
		return res.T
