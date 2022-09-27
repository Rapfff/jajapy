from sys import platform
from multiprocessing import cpu_count, Pool
from numpy import array, dot, append, zeros, ones, longdouble, float64
from datetime import datetime
from .Set import Set

NB_PROCESS = cpu_count()-1

class BW:
	"""
	Abstract class for general Baum-Welch algorithm.
	"""
	def __init__(self):
		pass

	def h_tau(self,s1: int,s2: int,obs: str) -> float:
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

	def computeAlphas(self, sequence: list, dtype=False) -> array:
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
				p = array([self.h_tau(ss,s,sequence[k]) for ss in range(self.nb_states)],dtype=dtype)
				alpha_matrix[k+1,s] = dot(alpha_matrix[k],p)
		return alpha_matrix.T

	def computeBetas(self, sequence: list, dtype=False) -> array:
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
				p = array([self.h_tau(s,ss,sequence[k]) for ss in range(self.nb_states)],dtype=dtype)
				beta_matrix[k,s] = dot(beta_matrix[k+1],p)
		return beta_matrix.T

	def _processWork(self,sequence,times):
		#overrided
		pass

	def _generateHhat(self):
		#overrided
		pass

	def _runProcesses(self,training_set):
		if platform != "win32" and platform != "darwin":
			p = Pool(processes = NB_PROCESS)
			tasks = []
			for seq,times in zip(training_set.sequences,training_set.times):
				tasks.append(p.apply_async(self._processWork, [seq, times,]))
			return [res.get() for res in tasks if res.get() != False]
		else:
			return [self._processWork(seq, times) for seq,times in zip(training_set.sequences,training_set.times)]
	
	def _endPrint(self,it,rt,ll):
		print()
		print("---------------------------------------------")
		print("Learning finished")
		print("Iterations:\t  ",it)
		print("Running time:\t  ",rt)
		print("---------------------------------------------")
		print()

	def fit(self,training_set: Set,initial_model,output_file: str,epsilon: float, max_it: int, pp: str, verbose: bool):
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : Set
			The training set.
		initial_model : Model
			The first hypothesis.
		output_file : str
			If set path file of the output model. Otherwise the output model
			will not be saved into a text file.
		epsilon : float
			The learning process stops when the difference between the
			loglikelihood of the training set under the two last hypothesis is
			lower than ``epsilon``. The lower this value the better the output,
			but the longer the running time.
		max_it: int
			Maximal number of iterations. The algorithm will stop after `max_it`
			iterations.
			Default is infinity.
		pp : str
			Will be printed at each iteration.
		verbose: bool
			Print or not a small recap at the end of the learning.

		Returns
		-------
		Model
			fitted model.
		"""
		start_time = datetime.now()
		self.h = initial_model
		self.nb_states = self.h.nb_states

		counter = 0
		prevloglikelihood = 10
		nb_traces = sum(training_set.times)
		while counter < max_it:
			print(pp, datetime.now(),counter, prevloglikelihood/nb_traces,end='\r')
			temp = self._runProcesses(training_set)
			self.hhat, currentloglikelihood = self._generateHhat(temp)
			counter += 1
			self.h = self.hhat
			if abs(prevloglikelihood - currentloglikelihood) < epsilon:
				break
			else:
				prevloglikelihood = currentloglikelihood
		
		running_time = datetime.now()-start_time

		if output_file:
			self.h.save(output_file)
		
		if verbose:
			self._endPrint(counter,running_time,currentloglikelihood)

		return self.h
