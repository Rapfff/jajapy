from ..base.tools import resolveRandom, randomProbabilities, checkProbabilities
from ..base.Model import Model
from ast import literal_eval
from numpy import ndarray, array, where, zeros

class HMM(Model):

	def __init__(self,matrix, output, alphabet, initial_state,name="unknown_HMM"):
		"""
		Creates an HMM.

		Parameters
		----------
		matrix : ndarray
			Represents the transition matrix.
			`matrix[s1][s2]` is the probability of moving from `s1` to `s2`.
		output : ndarray
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
		self.alphabet = alphabet
		self.output = output
		super().__init__(matrix,initial_state,name)
		for i in range(self.nb_states):
			if not checkProbabilities(output[i]):
				print("Error: the probability to generate an observation in state",i,"should be 1.0, here it's",output[i].sum())
				return False
			if not checkProbabilities(matrix[i]):
				print("Error: the probability to take a transition from state",i,"should be 1.0, here it's",matrix[i].sum())
				return False

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
		if s1 < 0 or s1 >= self.nb_states or s2 < 0 or s2 >= self.nb_states:
			return 0.0 
		return self.matrix[s1][s2]

	def b(self, s: int, l: str) -> float:
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
		return self.a(s1,s2)*self.b(s1,obs)
	
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
			return [self.alphabet[i] for i in where(self.output[state] > 0.0)[0]]

	def _stateToString(self, state: int) -> str:
		res = "----STATE s"+str(state)+"----\n"
		for j in range(len(self.matrix[state])):
			if self.matrix[state][j] > 0.0001:
				res += "s"+str(state)+" -> s"+str(j)+" : "+str(self.matrix[state][j])+'\n'
		res += "************\n"
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
		f.write(str(self.output.tolist()))
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

	def next_state(self, state: int) -> int:
		"""
		Returns one state according to the distribution described by the `self.next_matrix`.
		
		Returns
		-------
		int
			A state ID.
		"""
		c = resolveRandom(self.matrix[state].flatten())
		return c

	def next(self, state: int) -> list:
		"""
		Returns a state-observation pair according to the distributions described by `self.next_matrix` and `self.output_matrix`.

		Returns
		-------
		[int, str]
			A state-observation pair.
		"""
		return [self.next_state(state),self.next_obs(state)]

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


def HMM_random(nb_states: int, alphabet: list, random_initial_state: bool = False) -> HMM:
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
	return HMM(matrix, output, alphabet, init,"HMM_random_"+str(nb_states)+"_states")

def HMM_state(output:list, transitions:list, alphabet:list, nb_states:int) -> ndarray:
	"""
	Given the list of all transition leaving a state `s`, it generates
	the ndarray describing this state `s` in the HMM.matrix.
	This method is useful while creating a model manually.

	Parameters
	----------
	transitions : [ list of tuples (int, float)]
		Each tuple represents a transition as follow: 
		(destination state ID, probability).
	output : [ list of tuples (str, float)]
		Each tuple represents an output as follow: 
		(observation, probability).
	alphabet : list
		alphabet of the model in which this state is.
	nb_states: int
		number of states in which this state is

	Returns
	-------
	ndarray
		ndarray describing this state `s` in the HMM.matrix.
	"""

	res1 = zeros(nb_states)
	res2 = zeros(len(alphabet))
	for t in transitions:
		res1[t[0]] = t[1]
	for t in output:
		res2[alphabet.index(t[0])] = t[1]
	return [res1,res2]
