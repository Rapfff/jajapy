from ..base.tools import resolveRandom, randomProbabilities, checkProbabilities
from ..base.Base_HMM import Base_HMM,HMM_ID
from ast import literal_eval
from numpy import array, where, zeros
from numpy.random import seed

class HMM(Base_HMM):

	def __init__(self,matrix, output, alphabet, initial_state,name="unknown_HMM"):
		"""
		Creates an HMM.

		Parameters
		----------
		matrix : ndarray
			Represents the transition matrix.
			`matrix[s1][s2]` is the probability of moving from `s1` to `s2`.
		output : ndarray or list
			Represents the output matrix.
			`output[s1][obs]` is the probability of seeing `alphabet[obs]` in 
			state `s1`. 
		alphabet: list
			The list of all possible alphabet, such that:
			`alphabet.index("obs")` is the ID of `obs`.
		initial_state : int or list of float
			Determine which state is the initial one (then it's the id of the
			state), or what are the probability to start in each state (then it's
			a list of probabilities).
		name : str, optional
			Name of the model.
			Default is "unknow_MC"
		"""
		self.model_type = HMM_ID
		self.alphabet = alphabet
		self.output = array(output)
		super().__init__(matrix,initial_state,name)
		for i in range(self.nb_states):
			if not checkProbabilities(output[i]):
				raise ValueError("The probability to generate an observation in state",i,"should be 1.0, here it's",output[i].sum())

	def b(self, s: int, l: str) -> float:
		"""
		Returns the probability of generating `l` in state `s`.
		If `s` is not a valid state ID it returns 0.

		Parameters
		----------
		s : int
			ID of the source state.		
		l : str
			observation.
		
		Returns
		-------
		float
			probability of generating `l` in state `s`.

		Examples
		--------
		>>> model.b(0,'x')
		0.4
		>>> model.b(0,'foo')
		0.0
		"""
		if s < 0 or s >= self.nb_states or l not in self.alphabet:
			return 0.0 
		return self.output[s][self.alphabet.index(l)]
	
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
		return super().tau(s1,s2,obs)
	
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
			self._checkStateIndex(state)
			return [self.alphabet[i] for i in where(self.output[state] > 0.0)[0]]

	def _stateToString(self, state: int) -> str:
		res = super()._stateToString(state)
		for j in range(len(self.output[state])):
			if self.output[state][j] > 0.0001:
				res += "s"+str(state)+" => "+str(self.alphabet[j])+" : "+str(self.output[state][j])+'\n'
		return res
	
	def save(self,file_path:str):
		"""
		Save the model into a text file.

		Parameters
		----------
		file_path : str
			path of the output file.
		
		Examples
		--------
		>>> model.save("my_model.txt")
		"""
		f = open(file_path, 'w')
		f.write("HMM\n")
		f.write(str(self.alphabet))
		f.write('\n')
		super()._save(f)

	def next_obs(self, state: int) -> str:
		"""
		Generates one observation according to the distribution described by `self.output_matrix`.
		
		Returns
		-------
		str
			An observation.
		"""
		c = resolveRandom(self.output[state].flatten())
		return self.alphabet[c]


def loadHMM(file_path: str) -> HMM:
	"""
	Load an HMM saved into a text file.

	Parameters
	----------
	file_path : str
		Location of the text file.
	
	Returns
	-------
	output : HMM
		The HMM saved in `file_path`.
	"""
	f = open(file_path,'r')
	l = f.readline()[:-1] 
	if l != "HMM":
		print("ERROR: this file doesn't describe an HMM: it describes a "+l)
	alphabet = literal_eval(f.readline()[:-1])
	output = literal_eval(f.readline()[:-1])
	output = array(output)
	name = f.readline()[:-1]
	initial_state = array(literal_eval(f.readline()[:-1]))
	matrix = literal_eval(f.readline()[:-1])
	matrix = array(matrix)
	f.close()
	return HMM(matrix, output, alphabet, initial_state, name)


def HMM_random(nb_states: int, alphabet: list,
	    	   random_initial_state: bool = False,
			   sseed: int = None) -> HMM:
	"""
	Generates a random HMM.

	Parameters
	----------
	nb_states : int
		Number of states.
	alphabet : list of str
		List of observations.
	random_initial_state: bool, optional
		If set to True we will start in each state with a random probability, otherwise we will always start in state 0.
		Default is False.
	sseed : int, optional
		the seed value.
	
	Returns
	-------
	HMM
		A pseudo-randomly generated HMM.
	
	Examples
	--------
	>>> m = ja.HMM_random(4,['a','b','x','y'])
	"""
	matrix = []
	output = []
	if sseed != None:
		seed(sseed)
	for s in range(nb_states):
		p1 = array(randomProbabilities(nb_states))
		matrix.append(p1)
		p2 = array(randomProbabilities(len(alphabet)))
		output.append(p2)
	matrix = array(matrix)
	output = array(output)

	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	seed()
	return HMM(matrix, output, alphabet, init,"HMM_random_"+str(nb_states)+"_states")


def createHMM(transitions: list, emission: list, initial_state, name: str ="unknown_HMM") -> HMM:
	"""
	An user-friendly way to create a HMM.

	Parameters
	----------
	transitions : [ list of tuples (int, int, float)]
		Each tuple represents a transition as follow: 
		(source state ID, destination state ID, probability).
	emission : [ list of tuples (int, str, float)]
		Each tuple represents an emission probability as follow: 
		(source state ID, emitted label, probability).
	initial_state : int or list of float
		Determine which state is the initial one (then it's the id of the
		state), or what are the probability to start in each state (then it's
		a list of probabilities).
	name : str, optional
		Name of the model.
		Default is "unknow_HMM"
	
	Returns
	-------
	HMM
		the HMM describes by `transitions`, `emission`, and `initial_state`.
	
	Examples
	--------
	>>> model = createHMM([(0,1,1.0),(1,0,0.6),(1,1,0.4)],[(0,'a',0.8),(0,'b',0.2),(1,'b',1.0)],0,"My_HMM")
	>>> print(model)
	Name: My_HMM
	Initial state: s0
	----STATE s0----
	s0 -> s1 : 1.0
	************
	s0 => a : 0.8
	s0 => b : 0.2	
	----STATE s1----
	s1 -> s0 : 0.6
	s1 -> s1 : 0.4
	************
	s1 => b : 1.0
	"""
	
	states = list(set([i[0] for i in transitions]+[i[1] for i in transitions]))
	states.sort()
	alphabet = list(set([i[1] for i in emission]))
	nb_states = len(states)
	matrix = zeros((nb_states,nb_states))
	for t in transitions:
		matrix[states.index(t[0])][states.index(t[1])] = t[2]
	output = zeros((nb_states,len(alphabet)))
	for t in emission:
		output[states.index(t[0])][alphabet.index(t[1])] = t[2]
	return HMM(matrix, output, alphabet, initial_state, name)
