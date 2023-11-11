from .Scheduler import UniformScheduler
from ..base.BW import BW
from ..base.Set import Set
from ..base.tools import resolveRandom
from multiprocessing import Pool
from random import random
from numpy import zeros, dot, array, inf
from time import perf_counter

class ActiveLearningScheduler:
	"""
	Class for a scheduler used during the active learning sampling.
	"""
	def __init__(self,mat,m) -> None:
		"""
		Create an ActiveLearningScheduler.

		Parameters
		----------
		m : MDP
			The current hypothesis.
		"""
		self.m = m
		self.mat = mat
		self.nb_states = self.m.nb_states
		self.reset()
		
	def reset(self) -> None:
		"""
		Resets the scheduler. Should be done before genereting a new trace.
		"""
		self.last_act = None
		self.alpha_matrix = array([self.m.pi(s) for s in range(self.nb_states)])
	
	def getActionState(self,s:int) -> str:
		"""
		Returns an action chosen by the scheduler
		according to the expected current state.

		Parameters
		----------
		s : int
			Expected current state.

		Returns
		-------
		str
			Action chosen by the scheduler.
		"""
		tot = self.mat[s].sum()
		acts = [(tot-i)/(tot*(self.m.nb_actions-1)) for i in self.mat[s]]
		return resolveRandom(acts)


	def getAction(self) -> str:
		"""
		Returns the action chosen by the scheduler

		Returns
		-------
		str
			Action chosen by the scheduler.
		"""
		tot = self.alpha_matrix.sum()
		if tot <= 0.0:
			t = [1/len(self.alpha_matrix)]*len(self.alpha_matrix)
		else:
			t = [i/tot for i in self.alpha_matrix]
		s_i = resolveRandom(t)
		self.last_act = self.getActionState(s_i)
		return self.m.actions[self.last_act]


	def addObservation(self,obs: str) -> None:
		"""
		Updates the scheduler according to the given observation ``obs``.

		Parameters
		----------
		obs : str
			An observation.
		"""
		tot = sum(self.alpha_matrix)
		if tot > 0.0:
			for s in range(self.m.nb_states):
				self.mat[s,self.last_act] += self.alpha_matrix[s]/tot
			alpha_matrix = zeros(self.nb_states)
			for s in range(self.nb_states):
				p = array([self.m.tau(ss,self.m.actions[self.last_act],s,obs) for ss in range(self.nb_states)])
				alpha_matrix[s] = dot(self.alpha_matrix,p)
			self.alpha_matrix = alpha_matrix


class Active_BW_MDP(BW):
	"""
	Class for the Active Baum-Welch algorithm on MDP.
	This algorithm is described here:
	https://arxiv.org/pdf/2110.03014.pdf
	"""
	def __init__(self):
		super().__init__()


	def fit(self,traces, sul, lr, nb_iterations: int, nb_sequences: int,
			sequence_length: int = None, epsilon_greedy: float = 0.9,
			initial_model=None, nb_states: int=None,
			random_initial_state: bool=True, output_file: str=None,
			output_file_prism: str=None, epsilon: float=0.01, max_it: int=inf,
			pp: str='', verbose: int=2, return_data: bool=False,
			stormpy_output: bool = True, processes: int = None):
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : Set, list or ndarray.
			The training set.
		sul : MDP
			The system under learning.
			Used to generate new sequences.
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
		verbose: int, optional.
			Define the level of information that will print during the learning
			0 - nothing (no warnings, no progress bar, no recap at the end)
			1 - minimal (warnings only)
			2 - default (warnings and progress bar, no recap at the end)
			3 - maximal (warnings, progress bar and recap)
			Default is 2.
		return_data: bool, optional.
			If set to True, a dictionary containing following values will be
			returned alongside the output model once the learning is done.
			'learning_rounds', 'learning_time', 'training_set_loglikelihood'.
			Default is False.
		stormpy_output: bool, optional.
			If set to True the output model will be a Stormpy sparse model.
			Doesn't work for HMM and GOHMM.
			Default is True.
		processes : int, optional
			Number of processes used during the learning.
			Only for linux: for Windows and Mac OS it is 1.
			Default is `cpu_count()-1`.

		Returns
		-------
		MDP
			fitted MDP.
		"""
		if stormpy_output and not self.stormpy_installed:
			if self.verbose > 0:
				print("WARNING: stormpy not found. The output model will be a Jajapy model")
			stormpy_output = False

		counter = 0
		self.sul = sul
		start_time = perf_counter()
		
		_, info = super().fit(traces, initial_model=initial_model, nb_states=nb_states,
							  random_initial_state=random_initial_state, output_file=output_file,
							  output_file_prism=output_file_prism, epsilon=epsilon, max_it=max_it,
							  pp=pp+" Initial iteration:", verbose=verbose,
							  return_data=True, stormpy_output=False,
							  processes=processes)
		counter += info['learning_rounds']
	
		total_traces = traces
		
		if sequence_length == None:
			sequence_length = int(len(traces.sequences[0])/2)

		c = 1
		while c <= nb_iterations :
			traces = self._addTraces(sequence_length,nb_sequences,total_traces,epsilon_greedy)
			total_traces.addSet(traces)
			
			if lr == 0:
				_,info = super().fit(total_traces, initial_model=self.h,
							output_file=output_file, epsilon=epsilon,
							pp=pp+" Active iteration "+str(c)+"/"+str(nb_iterations)+": ",
							return_data=True, stormpy_output=False, verbose=verbose,processes=processes)
				counter += info['learning_rounds']
			else:
				if lr == "dynamic":
					lr_it = sum(traces.times)/(sum(total_traces.times)-sum(traces.times))
				else:
					lr_it = lr
				old_h = self.h
				_, info = super().fit(traces, initial_model=self.h,
							output_file=output_file, epsilon=epsilon,
							pp=pp+" Active iteration "+str(c)+"/"+str(nb_iterations)+": ",
							return_data=True, stormpy_output=False, verbose=verbose,processes=processes)
				counter += info['learning_rounds']
				self._mergeModels(old_h,lr_it)

			c += 1
		
		running_time = perf_counter()-start_time

		if output_file:
			self.h.save(output_file)
		
		if output_file_prism:
			self.h.savePrism(output_file_prism)
		
		if verbose == 3:
			self._endPrint(counter,running_time)

		if stormpy_output :
			self.h = self.jajapyModeltoStormpy(self.h)
			
		if return_data:
			info = {"learning_rounds":counter,
					"learning_time":running_time,
					"training_set_loglikelihood":self.h.logLikelihood(total_traces)}
			return self.h, info
	


		return self.h

	def _mergeModels(self,old_h,lr: float) -> None:
		"""
		Merges the model ``self.h`` with model ``old_h`` using the learning
		rate ``lr``.
		WARNING: we assume that self.h.alphabet == old_h.alphabet and
		self.h.actions == old_h.actions.

		Parameters
		----------
		old_h : MDP
			A MDP to merge with ``self.h``. 
		lr : float
			Learning rate. It is the weight of ``self.h`` in the output model.
		"""
		self.h.matrix = lr*self.h.matrix + (1-lr)*old_h.matrix

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
		mat = self._strategy(traces)
		scheduler_exploit = ActiveLearningScheduler(mat,self.h)
		scheduler_explore = UniformScheduler(self.h.getActions())

		traces = Set([],t=1)
		for n in range(nb_sequence):			
			if random() > epsilon_greedy:
				seq = self.sul.run(sequence_length,scheduler_explore)
			else:
				seq = self.sul.run(sequence_length,scheduler_exploit)

			if not seq in traces.sequences:
				traces.sequences.append(seq)
				traces.times.append(0)
			traces.times[traces.sequences.index(seq)] += 1
		return traces

	def _strategy(self,traces:Set) -> array:
		if self.processes > 1:
			p = Pool(processes = self.processes)
			tasks = []
			for seq,times in zip(traces.sequences,traces.times):
				tasks.append(p.apply_async(self._computeProbas, [seq, times,]))
			temp = [res.get() for res in tasks if res != False]
		else:
			temp = [self._computeProbas(seq, times) for seq,times in zip(traces.sequences,traces.times)]
			temp = [i for i in temp if i!=False]
		res = sum(temp)
		return res

	def _computeProbas(self,seq:list,times:int) -> array:
		sequence_actions = [seq[i+1] for i in range(0,len(seq)-1,2)]
		sequence_obs = [seq[i] for i in range(0,len(seq)-1,2)] + [seq[-1]]
		alpha_matrix = self._computeAlphas(sequence_obs,sequence_actions)
		beta_matrix  = self._computeBetas( sequence_obs,sequence_actions)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq <= 0.0:
			return False
		res = zeros(shape=(self.nb_states,len(self.actions)))

		for s in range(self.nb_states):
			for i,a in enumerate(self.actions):
				arr_dirak = array([1.0 if t == a else 0.0 for t in sequence_actions])
				res[s,i] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*arr_dirak,times/proba_seq).sum()
		return res
