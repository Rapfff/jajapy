from sys import platform
from multiprocessing import cpu_count, Pool
from numpy import array, dot, append, zeros, ones, float64, inf
from datetime import datetime
from .Set import Set
from alive_progress import alive_bar

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
	
	def _endPrint(self,it,rt):
		print()
		print("---------------------------------------------")
		print("Learning finished")
		print("Iterations:\t  ",it)
		print("Running time:\t  ",rt)
		print("---------------------------------------------")
		print()

	def fit(self,training_set: Set,initial_model,output_file: str,
			epsilon: float, max_it: int, pp: str, verbose: bool,
			return_data: bool, stormpy_output: bool = False):
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : Set
			The training set.
		initial_model : Model or stormpy.sparse model
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
		return_data: bool
			If set to True, a dictionary containing following values will be
			returned alongside the hypothesis once the learning is done.
			'learning_rounds', 'learning_time', 'training_set_loglikelihood'.
		stormpy_output: bool
			If set to True the output model will be a Stormpy sparse model.
			Doesn't work for GOHMM and MGOHMM.

		Returns
		-------
		Model
			fitted model.
		"""
		try:
			from ..with_stormpy import jajapyModeltoStorm, stormModeltoJajapy
			stormpy_installed = True
		except ModuleNotFoundError:
			stormpy_installed = False
		
		if stormpy_output and not stormpy_installed:
			print("WARNING: stormpy not found. The output model will not be a stormpy sparse model")
			stormpy_output = False
		
		try:
			initial_model.name
		except AttributeError: # then initial_model is a stormpy sparse model
			if not stormpy_installed:
				print("ERROR: the initial model is a Storm model and Storm is not installed on the machine")
				return False
			initial_model = stormModeltoJajapy(initial_model)		

		start_time = datetime.now()
		self.h = initial_model
		self.nb_states = self.h.nb_states

		counter = 0
		prevloglikelihood = 10

		if max_it != inf:
			alive_parameter = max_it
		else:
			alive_parameter = 0
		
		with alive_bar(alive_parameter, dual_line=True, title=str(pp)) as bar:
			while counter < max_it:
				temp = self._runProcesses(training_set)
				self.hhat, currentloglikelihood = self._generateHhat(temp)
				counter += 1
				self.h = self.hhat
				bar_text = '   Diff. loglikelihood: '+str(round(currentloglikelihood-prevloglikelihood,5))+' (>'+str(epsilon)+')'
				bar_text+= '   Av. one iteration (s): '+str(round((datetime.now()-start_time).total_seconds()/counter,2))
				bar.text = bar_text
				bar()
				if abs(prevloglikelihood - currentloglikelihood) < epsilon:
					break
				else:
					prevloglikelihood = currentloglikelihood

		running_time = datetime.now()-start_time
		running_time = running_time.total_seconds()

		if output_file:
			self.h.save(output_file)
		
		if verbose:
			self._endPrint(counter,running_time)

		if stormpy_output:
			self.h = jajapyModeltoStorm(self.h)

		if return_data:
			info = {"learning_rounds":counter,"learning_time":running_time,"training_set_loglikelihood":currentloglikelihood}
			return self.h, info
			
		return self.h
