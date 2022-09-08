from numpy.random import exponential
from ast import literal_eval
from ..base.tools import resolveRandom, normalize, randomProbabilities
from ..base.Set import Set
from ..mc.MC import MC, MC_state
from ..base.Model import Model
from math import exp, log
from random import randint
from numpy import array, zeros, dot
from sys import platform
from multiprocessing import cpu_count, Pool

class CTMC_state:
	"""
	Class for a CTMC state
	"""
	def __init__(self, lambda_matrix: list, idd: int) -> None:
		"""
		Creates a CTMC_state

		Parameters
		----------
		lambda_matrix : [ list of tuples (int, str, float)]
			Each tuple represents a transition as follow: 
			(destination state ID, observation, exit rate).
		idd : int
			State ID.
		"""
		states = [lambda_matrix[i][0] for i in range(len(lambda_matrix))]
		observations = [lambda_matrix[i][1] for i in range(len(lambda_matrix))]
		lambdaa = [lambda_matrix[i][2] for i in range(len(lambda_matrix))]
		if len(lambdaa) != 0:
			if min(lambdaa) < 0.0:
				print("Error: all rates should be strictly positive.")
		self.lambda_matrix = [lambdaa, states, observations]
		self.id = idd

	def tau(self, s: int, obs: str) -> float:
		"""
		Returns the probability to move from this state to state `s` generating
		observation ``obs``.

		Parameters
		----------
		s : int
			A state ID.
		obs : str
			An observation.

		Returns
		-------
		float
			A probability.
		"""
		return self.l(s,obs)/self.e()
	
	def l(self, s: int, obs: str) -> float:
		"""
		Returns the rate associated to the transition from this state to state
		``s`` generating observation ``obs``.

		Parameters
		----------
		s : int
			A state ID.
		obs : str
			An observation.

		Returns
		-------
		float
			A rate.
		"""
		for i in range(len(self.lambda_matrix[0])):
			if self.lambda_matrix[1][i] == s and self.lambda_matrix[2][i] == obs:
				return self.lambda_matrix[0][i]
		return 0.0

	def observations(self) -> list:
		"""
		Return the list of all the observations that can be generated from this state.

		Returns
		-------
		output : list of str
			A list of observations.
		"""
		return list(set(self.lambda_matrix[2]))

	def lkl(self,t: float) -> float:
		if t < 0.0:
			return 0.0
		"""
		Returns the likelihood of staying in this state for a duration ``t``.

		Parameters
		----------
		t : float
			A Waiting time.

		Returns
		-------
		float
			A Likelihood.
		"""
		return self.e()*exp(-self.e()*t)

	def e(self) -> float:
		"""
		Returns the exit rate of this state, i.e. the sum of all the rates in
		this state.

		Returns
		-------
		float
			An exit rate.
		"""
		return sum(self.lambda_matrix[0])

	def expected_time(self) -> float:
		"""
		Returns the expected waiting time in this state, i.e. the inverse of
		the exit rate.

		Returns
		-------
		float
			expected waiting time in this state.
		"""
		return 1/self.e()

	def next(self) -> list:
		"""
		Return a waiting time-state-observation tuple according to the rates
		in ``self.lambda_matrix``.

		Returns
		-------
		output : [float, int, str]
			A waiting time-state-observation tuple.
		"""
		exps = []
		for exp_lambda in self.lambda_matrix[0]:
			exps.append(exponential(1/exp_lambda))
		next_index = exps.index(min(exps))
		return [min(exps), self.lambda_matrix[1][next_index], self.lambda_matrix[2][next_index]]

	def __str__(self) -> str:
		res = "----STATE s"+str(self.id)+"----\n"
		res += "Exepected waiting time: "+str(self.expected_time())+'\n'
		
		den = sum(self.lambda_matrix[0])
		for j in range(len(self.lambda_matrix[0])):
			if self.lambda_matrix[0][j]/den > 0.0001:
				res += "s"+str(self.id)+" - ("+self.lambda_matrix[2][j]
				res += ") -> s"+str(self.lambda_matrix[1][j])+" : lambda = "+str(self.lambda_matrix[0][j])+'\n'
		return res

	def save(self) -> str:
		if len(self.lambda_matrix[0]) == 0: #end state
			return "-\n"
		else:
			res = ""
			for proba in self.lambda_matrix[0]:
				res += str(proba)+' '
			res += '\n'
			for state in self.lambda_matrix[1]:
				res += str(state)+' '
			res += '\n'
			for obs in self.lambda_matrix[2]:
				res += str(obs)+' '
			res += '\n'
			return res


class CTMC(Model):
	"""
	Class representing a CTMC.
	"""
	def __init__(self,states: list,initial_state,name: str="unknown_CTMC") -> None:
		"""
		Creates an CTMC.

		Parameters
		----------
		states : list of CTMC_states
			List of states in this CTMC.
		initial_state : int or list of float
			Determine which state is the initial one (then it's the id of the
			state), or what are the probability to start in each state (then it's
			a list of probabilities).
		name : str, optional
			Name of the model.
			Default is "unknow_CTMC"
		"""
		super().__init__(states,initial_state,name)

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
		return self.states[s].e()
	
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
		return self.states[s1].l(s2,obs)
	
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
		return self.states[s].lkl(t)

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
			[time_spent, next_state, symbol] = self.states[current].next()
			
			if timed:
				output.append(time_spent)

			output.append(symbol)
			current = next_state
			c += 1
		return output
	
	def _computeAlphas_timed(self, sequence: list, times: int) -> float:
		obs_seq   = [sequence[i] for i in range(1,len(sequence),2)]
		times_seq = [sequence[i] for i in range(0,len(sequence),2)]
		nb_states = len(self.states)
		len_seq = len(obs_seq)
		prev_arr = array(self.initial_state)
		for k in range(len_seq):
			new_arr = zeros(nb_states)
			for s in range(nb_states):
				p = array([self.l(ss,s,obs_seq[k])*exp(-self.e(ss)*times_seq[k]) for ss in range(nb_states)])
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

	def __str__(self) -> str:
		res = self.name+'\n'
		res += str(self.initial_state)+'\n'
		for i in range(len(self.states)):
			res += str(self.states[i])
		return res

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
		new_states = []
		for i in range(len(self.states)):
			s = self.states[i]
			den = sum(s.lambda_matrix[0]) 
			p = [s.lambda_matrix[0][j]/den for j in range(len(s.lambda_matrix[0]))]
			p = normalize(p)
			ss = s.lambda_matrix[1]
			o = s.lambda_matrix[2]

			new_states.append(MC_state([p,ss,o],i))

		return MC(new_states,self.initial_state,name)



def loadCTMC(file_path: str) -> CTMC:
	"""
	Load a model saved into a text file

	:param file_path: location of the text file
	:type file_path: str

	:return: a CTMC
	:rtype: CTMC
	"""
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(CTMC_state([[],[],[]]))
		else:
			p = [ float(i) for i in l[:-2].split(' ')]
			l = f.readline()[:-2].split(' ')
			s = [ int(i) for i in l ]
			o = f.readline()[:-2].split(' ')
			states.append(CTMC_state(list(zip(s,o,p)),len(states)))

		l = f.readline()

	return CTMC(states,initial_state,name)


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
	def computeFinalStateIndex(i1: int, i2: int, max2: int) -> int:
		return max2 * i1 + i2

	new_states = []
	initial_state = []
	max2 = len(m2.states)

	for i1 in range(len(m1.states)):
		s1 = m1.states[i1]
		
		for i2 in range(len(m2.states)):
			s2 = m2.states[i2]
			p = s1.lambda_matrix[0] + s2.lambda_matrix[0]
			s = [computeFinalStateIndex(s1.lambda_matrix[1][i],i2,max2) for i in range(len(s1.lambda_matrix[1]))]
			s+= [computeFinalStateIndex(i1,s2.lambda_matrix[1][i],max2) for i in range(len(s2.lambda_matrix[1]))]
			if not disjoint:
				o = s1.lambda_matrix[2] + s2.lambda_matrix[2]
			else:
				o = [i+'1' for i in s1.lambda_matrix[2]] + [i+'2' for i in s2.lambda_matrix[2]]
				
			initial_state.append(m1.initial_state[i1]*m2.initial_state[i2])
			new_states.append(CTMC_state(list(zip(s,o,p)),i1*len(m2.states)+i2))

	return CTMC(new_states,initial_state,name)

def CTMC_random(number_states: int, alphabet: list, min_exit_rate_time : int,
				max_exit_rate_time: int, self_loop: bool = True,
				random_initial_state: bool=False) -> CTMC:
	"""
	Generates a random CTMC. All the rates will be between 0 and 1.
	All the exit rates will be integers.

	Parameters
	----------
	number_states : int
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
	for j in range(number_states):
		s.append([])
		for i in range(number_states):
			if self_loop or i != j:
				s[j] += [i] * len(alphabet)
	if self_loop:
		obs = alphabet*number_states
	else:
		obs = alphabet*(number_states-1)

	states = []
	for i in range(number_states):
		random_probs = randomProbabilities(len(obs))
		av_waiting_time = randint(min_exit_rate_time,max_exit_rate_time)
		p = [x/av_waiting_time for x in random_probs]
		states.append(CTMC_state(list(zip(s[i],obs,p)),i))
	if random_initial_state:
		init = randomProbabilities(number_states)
	else:
		init = 0
	return CTMC(states,init,"CTMC_random_"+str(number_states)+"_states")