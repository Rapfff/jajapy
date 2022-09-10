from numpy.random import exponential
from ast import literal_eval
from ..base.tools import resolveRandom, normalize, randomProbabilities
from ..base.Set import Set
from ..mc.MC import MC
from ..base.Model import Model
from math import exp, log
from random import randint
from numpy import array, zeros, dot, ndarray, where, reshape
from sys import platform
from multiprocessing import cpu_count, Pool

class CTMC(Model):
	"""
	Class representing a CTMC.
	"""
	def __init__(self, matrix: ndarray, alphabet: list, initial_state, name: str ="unknown_MC") -> None:
		"""
		Creates an CTMC.

		Parameters
		----------
		matrix : ndarray
			Represents the transition matrix.
			`matrix[s1][s2][obs_ID]` is the rate associated to the transition 
			from `s1` to `s2` seeing the observation of ID `obs_ID`.
		alphabet: list
			The list of all possible observations, such that:
			`alphabet.index("obs")` is the ID of `obs`.
		initial_state : int or list of float
			Determine which state is the initial one (then it's the id of the
			state), or what are the probability to start in each state (then it's
			a list of probabilities).
		name : str, optional
			Name of the model.
			Default is "unknow_CTMC"
		"""
		self.alphabet = alphabet
		super().__init__(matrix,initial_state,name)

	def e(self,s: int) -> float:
		"""
		Returns the exit rate of state ``s``, i.e. the sum of all the rates in
		this state.

		Returns
		-------
		s : int
			A state ID.
		float
			An exit rate.
		"""
		return sum(self.matrix[s].flatten())
	
	def l(self, s1:int, s2:int, obs:str) -> float:
		"""
		Returns the rate associated to the transition from state `s1` to state
		``s2`` generating observation ``obs``.

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
		return self.matrix[s1][s2][self.alphabet.index(obs)]
	
	def lkl(self, s: int, t: float) -> float:
		"""
		Returns the likelihood of staying in `s` state for a duration ``t``.

		Parameters
		----------
		s : int
			A state ID.
		t : float
			A waiting time.

		Returns
		-------
		float
			A Likelihood.
		"""
		if t < 0.0:
			return 0.0
		return self.e(s)*exp(-self.e(s)*t)
	
	def tau(self, s1:int, s2: int, obs: str) -> float:
		"""
		Returns the probability to move from `s1` to `s2` generating
		observation ``obs``.

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
			A probability.
		"""
		return self.l(s1,s2,obs)/self.e(s1)
	
	def expected_time(self, s:int) -> float:
		"""
		Returns the expected waiting time in `s`, i.e. the inverse of
		the exit rate.
		
		Parameters
		----------
		s : int
			A state ID.
		
		Returns
		-------
		float
			expected waiting time in this state.
		"""
		return 1/self.e(s)

	def getAlphabet(self,state: int = -1) -> list:
		"""
		If state is set, returns the list of all the observations we could
		see in `state`. Otherwise it returns the alphabet of the model. 

		Parameters
		----------
		state : int, optional
			a state ID

		Returns
		-------
		list of str
			list of observations
		"""
		if state == -1:
			return self.alphabet
		else:
			return [self.alphabet[i] for i in where(self.matrix[state].sum(axis=0) > 0.0)[0]]
		
	def save(self,file_path:str):
		"""Save the model into a text file.

		Parameters
		----------
		file_path : str
			path of the output file.
		
		Examples
		--------
		>>> model.save("my_model.txt")
		"""
		f = open(file_path, 'w')
		f.write("CTMC\n")
		f.write(str(self.alphabet))
		f.write('\n')
		super()._save(f)

	def _stateToString(self,state:int) -> str:
		res = "----STATE s"+str(state)+"----\n"
		res += "Exepected waiting time: "+str(self.expected_time(state))+'\n'
		den = self.e(state)
		for j in range(len(self.matrix[state])):
			for k in range(len(self.matrix[state][j])):
				if self.matrix[state][j][k]/den > 0.0001:
					res += "s"+str(state)+" - ("+str(self.alphabet[k])+") -> s"
					res += str(j)+" : lambda = "+str(self.matrix[state][j][k])
					res += '\n'
		return res
	
	def next(self,state: int) -> tuple:
		"""
		Return a state-observation pair according to the distributions 
		described by matrix
		Returns
		-------
		output : (int, str)
			A state-observation pair.
		"""
		exps = []
		for exp_lambda in self.matrix[state].flatten():
			if exp_lambda <= 0.0:
				exps.append(1024)
			else:
				exps.append(exponential(1/exp_lambda))
		next_index = exps.index(min(exps))
		next_state = next_index//len(self.alphabet)
		next_obs = self.alphabet[next_index % len(self.alphabet)]
		return (next_state, next_obs, min(exps))

	def run(self,number_steps: int, timed: bool = False) -> list:
		"""
		Simulates a run of length ``number_steps`` of the model and return the
		sequence of observations generated. If ``timed`` it returns a list of
		pairs waiting time-observation.
		
		Parameters
		----------
		number_steps: int
			length of the simulation.
		timed: bool, optional
			Wether or not it returns also the waiting times. Default is False.

		Returns
		-------
		output: list of str
			trace generated by the run.
		"""
		output = []
		current = resolveRandom(self.initial_state)
		c = 0
		while c < number_steps:
			[next_state, symbol, time_spent] = self.next(current)
			
			if timed:
				output.append(time_spent)

			output.append(symbol)
			current = next_state
			c += 1
		return output
	
	def _computeAlphas_timed(self, sequence: list, times: int) -> float:
		obs_seq   = [sequence[i] for i in range(1,len(sequence),2)]
		times_seq = [sequence[i] for i in range(0,len(sequence),2)]
		len_seq = len(obs_seq)
		prev_arr = array(self.initial_state)
		for k in range(len_seq):
			new_arr = zeros(self.nb_states)
			for s in range(self.nb_states):
				p = array([self.l(ss,s,obs_seq[k])*exp(-self.e(ss)*times_seq[k]) for ss in range(self.nb_states)])
				new_arr[s] = dot(prev_arr,p)
			prev_arr = new_arr
		return log(prev_arr.sum())*times

	def logLikelihood(self,traces: Set) -> float:
		if type(traces.sequences[0][0]) == str: # non-timed traces
			return super().logLikelihood(traces)
		else: # timed traces
			if platform != "win32":
				p = Pool(processes = cpu_count()-1)
				tasks = []
				for seq,times in zip(traces.sequences,traces.times):
					tasks.append(p.apply_async(self._computeAlphas_timed, [seq, times,]))
				temp = [res.get() for res in tasks if res.get() != False]
			else:
				temp = [self._computeAlphas_timed(traces.sequences[i],traces.times[i]) for i in range(len(traces.sequences))]
			return sum(temp)/sum(traces.times)

	def toMC(self, name: str="unknown_MC") -> MC:
		"""
		Returns the equivalent untimed MC.

		Parameters
		----------
		name : str, optional
			Name of the output model. By default "unknown_MC"

		Returns
		-------
		MC
			An equivalent untimed model.
		"""
		new_matrix = self.matrix
		for i in range(self.nb_states):
			new_matrix[i] /= self.e(i)

		return MC(new_matrix,self.alphabet,self.initial_state,name)


def loadCTMC(file_path: str) -> MC:
	"""
	Load an CTMC saved into a text file.

	Parameters
	----------
	file_path : str
		Location of the text file.
	
	Returns
	-------
	output : CTMC
		The CTMC saved in `file_path`.
	"""
	f = open(file_path,'r')
	l = f.readline()[:-1] 
	if l != "CTMC":
		print("ERROR: this file doesn't describe an CTMC: it describes a "+l)
	alphabet = literal_eval(f.readline()[:-1])
	name = f.readline()[:-1]
	initial_state = array(literal_eval(f.readline()[:-1]))
	matrix = literal_eval(f.readline()[:-1])
	matrix = array(matrix)
	f.close()
	return CTMC(matrix, alphabet, initial_state, name)


def asynchronousComposition(m1: CTMC, m2: CTMC, name: str='unknown_composition', disjoint: bool=False) -> CTMC:
	"""
	Returns a CTMC equivalent to the asynchronous composition of `m1` and `m2`.

	Parameters
	----------
	m1 : CTMC
		First model of the composition.
	m2 : CTMC
		Second model of the composition.
	name : str, optional
		Name of the output model, by default 'unknown_composition'
	disjoint : bool, optional
		If True the observation generated by the composition contain 1 or 2
		according to which model generated this observation. By default False.

	Returns
	-------
	CTMC
		A CTMC equivalent to the asynchronous composition of `m1` and `m2`.

	"""

	nb_states = m1.nb_states * m2.nb_states
	if disjoint:
		alphabet = [i+'1' for i in m1.alphabet] + [i+'2' for i in m2.alphabet]
	else:
		alphabet = list(set(m1.alphabet).union(set(m2.alphabet)))

	matrix = zeros((nb_states, nb_states, len(alphabet)))
	initial_state = zeros(nb_states)
	for i in range(m1.nb_states):
		for j in range(m2.nb_states):
			initial_state[i*m2.nb_states+j] = m1.initial_state[i]*m2.initial_state[j]

	for s in range(nb_states):
		for s1 in range(m1.nb_states):
			dest = s1*m2.nb_states+(s%m2.nb_states)
			for i,o in enumerate(m1.alphabet):
				if disjoint:
					matrix[s][dest][i] = m1.matrix[s//m2.nb_states][s1][i]
				else:
					matrix[s][dest][alphabet.index(o)] = m1.matrix[s//m2.nb_states][s1][i]
		for s2 in range(m2.nb_states):
			dest = s2+s-(s%m2.nb_states)
			for i,o in enumerate(m2.alphabet):
				if disjoint:
					matrix[s][dest][len(m1.alphabet)+i] = m2.matrix[s%m2.nb_states][s2][i]
				else:
					matrix[s][dest][alphabet.index(o)] = m2.matrix[s%m2.nb_states][s2][i]

	return CTMC(matrix,alphabet,initial_state,name)

def CTMC_random(nb_states: int, alphabet: list, min_exit_rate_time : int,
				max_exit_rate_time: int, self_loop: bool = True,
				random_initial_state: bool=False) -> CTMC:
	"""
	Generates a random CTMC. All the rates will be between 0 and 1.
	All the exit rates will be integers.

	Parameters
	----------
	nb_states : int
		Number of states.
	alphabet : list of str
		List of observations.
	min_exit_rate_time: int
		Minimum exit rate for the states.
	max_exit_rate_time: int
		Minimum exit rate for the states.
	self_loop: bool, optional
		Wether or not there will be self loop in the output model.
		Default is True.
	random_initial_state: bool, optional
		If set to True we will start in each state with a random probability, otherwise we will always start in state 0.
		Default is False.

	Returns
	-------
	CTMC
		A pseudo-randomly generated CTMC.
	"""
	#lambda between 0 and 1
	s = []
	for j in range(nb_states):
		s.append([])
		for i in range(nb_states):
			if self_loop or i != j:
				s[j] += [i] * len(alphabet)
	if self_loop:
		obs = alphabet*nb_states
	else:
		obs = alphabet*(nb_states-1)

	matrix = []
	for i in range(nb_states):
		if self_loop:
			random_probs = array(randomProbabilities(len(alphabet)*nb_states))
		else:
			p = randomProbabilities(len(alphabet)*(nb_states-1))
			random_probs = array(p[:i*len(alphabet)]+[0.0 for _ in range(len(alphabet))]+p[i*len(alphabet):])
		av_waiting_time = randint(min_exit_rate_time,max_exit_rate_time)
		p = random_probs/av_waiting_time
		p = reshape(p, (nb_states,len(alphabet)))
		matrix.append(p)
	matrix = array(matrix)

	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return CTMC(matrix, alphabet, init,"CTMC_random_"+str(nb_states)+"_states")

def CTMC_state(transitions:list, alphabet:list, nb_states:int) -> ndarray:
	"""
	Given the list of all transition leaving a state `s`, it generates
	the ndarray describing this state `s` in the CTMC.matrix.
	This method is useful while creating a model manually.

	Parameters
	----------
	transitions : [ list of tuples (int, str, float)]
		Each tuple represents a transition as follow: 
		(destination state ID, observation, rate).
	alphabet : list
		alphabet of the model in which this state is.
	nb_states: int
		number of states in which this state is

	Returns
	-------
	ndarray
		ndarray describing this state `s` in the CTMC.matrix.
	"""

	res = zeros((nb_states,len(alphabet)))
	for t in transitions:
		res[t[0]][alphabet.index(t[1])] = t[2]
	return res
