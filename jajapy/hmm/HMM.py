from ..base.tools import resolveRandom, randomProbabilities
from ..base.Model import Model, Model_state
from ast import literal_eval

class HMM_state(Model_state):
	"""
	Creates a HMM_state

	Parameters
	----------
	next_matrix : [ list of tuples (int, float)]
		Each tuple represents a transition as follow: 
		(destination state ID, probability).
	output_matrix : [ list of tuples (str, float)]
		Each tuple represents an observation generation as follow: 
		(observation, probability).
	idd : int
		State ID.
	"""
	def __init__(self,output_matrix: list, next_matrix: list, idd: int):
		probabilities = [i[1] for i in output_matrix]
		observations = [i[0] for i in output_matrix]
		output_matrix = [probabilities, observations]
		states = [i[0] for i in next_matrix]
		probabilities = [i[1] for i in next_matrix]
		next_matrix = [probabilities, states]
		super().__init__(next_matrix, idd)
		self._checkTransitionMatrix(output_matrix)
		self.output_matrix = output_matrix

	def a(self, s: int) -> float:
		"""
		Returns the probability of moving, from this state, to state `s`.

		Parameters
		----------
		s : int
			The destination state ID.

		Returns
		-------
		float
			The probability of moving, from this state, to state `s`.
		"""
		if s in self.transition_matrix[1]:
			return self.transition_matrix[0][self.transition_matrix[1].index(s)]
		else:
			return 0.0

	def b(self, l: str) -> float:
		"""
		Returns the probability of generating, from this state, observation `l`.

		Parameters
		----------
		l : str
			The observation.

		Returns
		-------
		float
			The probability of generating, from this state, observation `l`.
		"""
		if l in self.output_matrix[1]:
			return self.output_matrix[0][self.output_matrix[1].index(l)]
		else:
			return 0.0

	def next_obs(self) -> str:
		"""
		Generates one observation according to the distribution described by `self.output_matrix`.
		
		Returns
		-------
		str
			An observation.
		"""
		return self.output_matrix[1][resolveRandom(self.output_matrix[0])]

	def next_state(self) -> int:
		"""
		Returns one state according to the distribution described by the `self.next_matrix`.
		
		Returns
		-------
		int
			A state ID.
		"""
		return self.transition_matrix[1][resolveRandom(self.transition_matrix[0])]

	def next(self) -> list:
		"""
		Returns a state-observation pair according to the distributions described by `self.next_matrix` and `self.output_matrix`.

		Returns
		-------
		[int, str]
			A state-observation pair.
		"""
		return [self.next_state(),self.next_obs()]
	
	def tau(self,s: int, obs: str) -> float:
		"""
		Returns the probability of generating, from this state, observation `obs` while moving to state `s`.

		Parameters
		----------
		s : int
			A state ID.
		obs : str
			An observation.

		Returns
		-------
		float
			The probability of generating, from this state, observation `obs` and moving to state `s`.
		"""
		return self.a(s)*self.b(obs)

	def observations(self) -> list:
		"""
		Returns the list of all the observations that can be generated from this state.

		Returns
		-------
		list of str
			A list of observations.
		"""
		return list(set(self.output_matrix[1]))
		

	def __str__(self) -> str:
		res = "----STATE s"+str(self.id)+"----\n"
		for j in range(len(self.transition_matrix[0])):
			if self.transition_matrix[0][j] > 0.0001:
				res += "s"+str(self.id)+" -> s"+str(self.transition_matrix[1][j])+" : "+str(self.transition_matrix[0][j])+'\n'
		res += "************\n"
		for j in range(len(self.output_matrix[0])):
			if self.output_matrix[0][j] > 0.0001:
				res += "s"+str(self.id)+" => "+str(self.output_matrix[1][j])+" : "+str(self.output_matrix[0][j])+'\n'
		return res

	def save(self) -> str:
		if len(self.transition_matrix[0]) == 0: #end state
			return "-\n"
		else:
			res = ""
			for proba in self.transition_matrix[0]:
				res += str(proba)+' '
			res += '\n'
			for state in self.transition_matrix[1]:
				res += str(state)+' '
			res += '\n'

			for proba in self.output_matrix[0]:
				res += str(proba)+' '
			res += '\n'
			for obs in self.output_matrix[1]:
				res += str(obs)+' '
			res += '\n'
			return res

class HMM(Model):
	"""
	Creates an HMM.

	Parameters
	----------
	states : list of HMM_states
		List of states in this HMM.
	initial_state : int or list of float
		Determine which state is the initial one (then it's the id of the
		state), or what are the probability to start in each state (then it's
		a list of probabilities).
	name : str, optional
		Name of the model.
		Default is "unknow_HMM"
	"""
	def __init__(self,states,initial_state,name="unknown_HMM"):
		super().__init__(states,initial_state,name)

	def a(self,s1: int,s2: int) -> float:
		"""
		Returns the probability of moving from state `s1` to state `s2`.
		If `s1` or `s2` is not a valid state ID it returns 0.

		Parameters
		----------
		s1 : int
			ID of the source state.		
		s2 : int
			ID of the destination state.
		
		Returns
		-------
		float
			Probability of moving from state `s1` to state `s2`.
		
		Examples
		--------
		>>> model.a(0,1)
		0.5
		>>> model.a(0,0)
		0.0
		"""
		if s1 < 0 or s1 >=len(self.states) or s2 < 0 or s2 >= len(self.states):
			return 0.0 
		return self.states[s1].a(s2)

	def b(self,s: int, l: str) -> float:
		"""
		Returns the probability of generating `l` in state `s`.
		If `s` is not a valid state ID it returns 0.

		Parameters
		----------
		s : int
			ID of the source state.		
		o : str
			observation.
		
		Returns
		-------
		float
			probability of generating `o` in state `s`.

		Examples
		--------
		>>> model.b(0,'x')
		0.4
		>>> model.b(0,'foo')
		0.0
		"""
		if s < 0 or s >=len(self.states):
			return 0.0 
		return self.states[s].b(l)
	
	def tau(self, s1: int, s2: int, obs: str) -> float:
		"""
		Return the probability of generating from state `s1` observation `obs` and moving to state `s2`.
		If `s1` or `s2` is not a valid state ID it returns 0.

		Parameters
		----------
		s1 : int
			ID of the source state.
		s2 : int
			ID of the destination state.
		obs : str
			An observation.

		Returns
		-------
		float
			The probability of generating from state `s1` observation `obs` and moving to state `s`.
		
		Examples
		--------
		>>> model.tau(0,1,'x')
		0.2
		"""
		if s1 < 0 or s1 >=len(self.states) or s2 < 0 or s2 >= len(self.states):
			return 0.0 
		return self.states[s1].tau(s2,obs)
	

def loadHMM(file_path: str) -> HMM:
	"""
	Loads an HMM saved into a text file.

	Parameters
	----------
	file_path : str
		Location of the text file.
	
	Returns
	-------
	HMM
		The HMM saved in `file_path`.
	
	Examples
	--------
	>>> model.save("test_save.txt")
	>>> mprime = ja.loadHMM("test_save.txt")
	>>> assert str(m) == str(mprime)
	"""
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	c = 0
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(HMM_state([[[],[]],[[],[]],c]))
		else:
			ps = [ float(i) for i in l[:-2].split(' ')]
			l  = f.readline()[:-2].split(' ')
			s  = [ int(i) for i in l ]
			l  = f.readline()[:-2].split(' ')
			po = [ float(i) for i in l]
			o  = f.readline()[:-2].split(' ')
			states.append(HMM_state([(o[i],po[i]) for i in range(len(o))],
									[(s[i],ps[i]) for i in range(len(s))],c))
		c += 1
		l = f.readline()

	return HMM(states,initial_state,name)

def HMM_random(number_states: int, alphabet: list, random_initial_state: bool = False) -> HMM:
	"""
	Generates a random HMM.

	Parameters
	----------
	number_states : int
		Number of states.
	alphabet : list of str
		List of observations.
	random_initial_state: bool, optional
		If set to True we will start in each state with a random probability, otherwise we will always start in state 0.
		Default is False.
	
	Returns
	-------
	HMM
		A pseudo-randomly generated HMM.
	
	Examples
	--------
	>>> m = ja.HMM_random(4,['a','b','x','y'])
	"""
	states = []
	for c in range(number_states):
		ps = randomProbabilities(number_states)
		po = randomProbabilities(len(alphabet))
		o = alphabet
		states.append(HMM_state([(po[i],o[i]) for i in range(len(o))],
									[(ps[i],i) for i in range(number_states)],c))
	if random_initial_state:
		init = randomProbabilities(number_states)
	else:
		init = 0
	return HMM(states,init)