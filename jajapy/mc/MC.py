from ..base.tools import resolveRandom, randomProbabilities, checkProbabilities
from ..base.Model import Model
from ast import literal_eval
from numpy import ndarray, array, where, reshape, zeros

class MC(Model):
	def __init__(self,matrix: ndarray, alphabet: list, initial_state, name: str ="unknown_MC") -> None:
		"""
		Creates an MC.

		Parameters
		----------
		matrix : ndarray
			Represents the transition matrix.
			`matrix[s1][s2][obs_ID]` is the probability of moving from `s1` to 
			`s2` seeing the observation of ID `obs_ID`.
		alphabet: list
			The list of all possible observations, such that:
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
		super().__init__(matrix,initial_state,name)
		for i in range(self.nb_states):
			if not checkProbabilities(matrix[i]):
				print("Error: the probability to take a transition from state",i,"should be 1.0, here it's",matrix[i].sum())
				return False
	
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

	def tau(self,s1: int,s2: int,obs) -> float:
		"""
		Returns the probability of moving from state ``s1`` to ``s2`` generating observation ``obs``.
		Parameters
		----------
		s1: int
			source state ID.
		s2: int
			destination state ID.
		obs: str
			generated observation.
		
		Returns
		-------
		float
			probability of moving from state ``s1`` to ``s2`` generating observation ``obs``.
		"""
		if not obs in self.alphabet:
			return 0.0
		return self.matrix[s1][s2][self.alphabet.index(obs)]

	def next(self,state: int) -> tuple:
		"""
		Return a state-observation pair according to the distributions 
		described by matrix
		Returns
		-------
		output : (int, str)
			A state-observation pair.
		"""
		c = resolveRandom(self.matrix[state].flatten())
		return (c//len(self.alphabet), self.alphabet[c%len(self.alphabet)])

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
		f.write("MC\n")
		f.write(str(self.alphabet))
		f.write('\n')
		super()._save(f)

	def _stateToString(self,state:int) -> str:
		res = "----STATE s"+str(state)+"----\n"
		for j in range(len(self.matrix[state])):
			for k in range(len(self.matrix[state][j])):
				if self.matrix[state][j][k] > 0.0001:
					res += "s"+str(state)+" - ("+str(self.alphabet[k])+") -> s"+str(j)+" : "+str(self.matrix[state][j][k])+'\n'
		return res

def loadMC(file_path: str) -> MC:
	"""
	Load an MC saved into a text file.

	Parameters
	----------
	file_path : str
		Location of the text file.
	
	Returns
	-------
	output : MC
		The MC saved in `file_path`.
	"""
	f = open(file_path,'r')
	l = f.readline()[:-1] 
	if l != "MC":
		print("ERROR: this file doesn't describe an MC: it describes a "+l)
	alphabet = literal_eval(f.readline()[:-1])
	name = f.readline()[:-1]
	initial_state = array(literal_eval(f.readline()[:-1]))
	matrix = literal_eval(f.readline()[:-1])
	matrix = array(matrix)
	f.close()
	return MC(matrix, alphabet, initial_state, name)

def MC_random(nb_states: int, alphabet: list, random_initial_state: bool=False) -> MC:
	"""
	Generate a random MC.

	Parameters
	----------
	number_states : int
		Number of states.
	alphabet : list of str
		List of alphabet.
	random_initial_state: bool, optional
		If set to True we will start in each state with a random probability, otherwise we will always start in state 0.
		Default is False.
	
	Returns
	-------
	A pseudo-randomly generated MC.
	"""
	matrix = []
	for s in range(nb_states):
		p = array(randomProbabilities(nb_states*len(alphabet)))
		p = reshape(p, (nb_states,len(alphabet)))
		matrix.append(p)
	matrix = array(matrix)

	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return MC(matrix, alphabet, init,"MC_random_"+str(nb_states)+"_states")

def MC_state(transitions:list, alphabet:list, nb_states:int) -> ndarray:
	"""
	Given the list of all transition leaving a state `s`, it generates
	the ndarray describing this state `s` in the MC.matrix.
	This method is useful while creating a model manually.

	Parameters
	----------
	transitions : [ list of tuples (int, str, float)]
		Each tuple represents a transition as follow: 
		(destination state ID, observation, probability).
	alphabet : list
		alphabet of the model in which this state is.
	nb_states: int
		number of states in which this state is

	Returns
	-------
	ndarray
		ndarray describing this state `s` in the MC.matrix.
	"""

	res = zeros((nb_states,len(alphabet)))
	for t in transitions:
		res[t[0]][alphabet.index(t[1])] = t[2]
	return res
