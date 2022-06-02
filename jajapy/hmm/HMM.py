from ..base.tools import resolveRandom, randomProbabilities
from ..base.Model import Model, Model_state
from ast import literal_eval

class HMM_state(Model_state):
	"""
	Creates a HMM_state

	Parameters
	----------
	output_matrix : [ list of float, list of str]
		`[[proba_symbol1,proba_symbol2,...],[symbol1,symbol2,...]]`. `output_matrix[0][x]` is the probability to generate the observation `output_matrix[1][x]`.
	next_matrix : [ list of float, list of int]
		`[[proba_state1,proba_state2,...],[state1,state2,...]]`. `next_matrix[0][x]` is the probability to move to state `next_matrix[1][x]`.
	idd : int
		State ID.
	"""
	def __init__(self,output_matrix: list, next_matrix: list, idd: int):
		super().__init__(next_matrix, idd)
		if round(sum(output_matrix[0]),2) != 1.0 and sum(output_matrix[0]) != 0:
			print("Sum of the probabilies of the output_matrix should be 1 or 0 here it's ",sum(output_matrix[0]))
			return False
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

		Parameters
		----------
		s1 : int
			ID of the source state.		
		s2 : int
			ID of the destination state.
		
		Returns
		-------
		float
			Probability of moving from state `s1` to state `s2`
		"""
		return self.states[s1].a(s2)

	def b(self,s: int, l: str) -> float:
		"""
		Returns the probability of generating `l` in state `s`.

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
		"""
		return self.states[s].b(l)
	
	def tau(self, s1: int, s2: int, obs: str) -> float:
		"""
		Return the probability of generating from state `s1` observation `obs` and moving to state `s2`.

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
		"""
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
			states.append(HMM_state([po,o],[ps,s],c))
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
	"""
	states = []
	for s in range(number_states):
		states.append(HMM_state([randomProbabilities(len(alphabet)),alphabet],[randomProbabilities(number_states),list(range(number_states))],s))

	if random_initial_state:
		init = randomProbabilities(number_states)
	else:
		init = 0
	return HMM(states,init)