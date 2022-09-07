from ..base.tools import resolveRandom, randomProbabilities
from ..base.Model import Model, Model_state
from ast import literal_eval

class MC_state(Model_state):
	"""
	Create a MC_state

	Parameters
	----------
	next_matrix : [ list of tuples (int, str, float)]
		Each tuple represents a transition as follow: 
		(destination state ID, observation, probability).
	idd : int
		State ID.
	"""

	def __init__(self,transitions: list, idd: int):
		probabilities = [i[2] for i in transitions]
		states = [i[0] for i in transitions]
		observations = [i[1] for i in transitions]
		next_matrix = [probabilities, states, observations]
		super().__init__(next_matrix,idd)

	def next(self) -> list:
		"""
		Return a state-observation pair according to the distributions 
		described by next_matrix

		Returns
		-------
		output : [int, str]
			A state-observation pair.
		"""
		c = resolveRandom(self.transition_matrix[0])
		return [self.transition_matrix[1][c],self.transition_matrix[2][c]]

	def tau(self,state: int, obs: str) -> float:
		"""
		Return the probability of generating, from this state, observation
		`obs` while moving to state `state`.

		Parameters
		----------
		state : int
			A state ID.
		obs : str
			An observation.

		Returns
		-------
		output : float
			A probability.
		"""
		for i in range(len(self.transition_matrix[0])):
			if self.transition_matrix[1][i] == state and self.transition_matrix[2][i] == obs:
				return self.transition_matrix[0][i]
		return 0.0

	def observations(self) -> list:
		"""
		Return the list of all the observations that can be generated from this state.

		Returns
		-------
		output : list of str
			A list of observations.
		"""
		return list(set(self.transition_matrix[2]))

	def __str__(self) -> str:
		res = "----STATE s"+str(self.id)+"----\n"
		for j in range(len(self.transition_matrix[0])):
			if self.transition_matrix[0][j] > 0.0001:
				res += "s"+str(self.id)+" - ("+str(self.transition_matrix[2][j])+") -> s"+str(self.transition_matrix[1][j])+" : "+str(self.transition_matrix[0][j])+'\n'
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
			for obs in self.transition_matrix[2]:
				res += str(obs)+' '
			res += '\n'
			return res


class MC(Model):
	"""
	Class representing a MC.
	"""	
	def __init__(self,states: list, initial_state: int ,name: str ="unknown_MC") -> None:
		"""
		Creates an MC.

		Parameters
		----------
		states : list of MC_states
			List of states in this MC.
		initial_state : int or list of float
			Determine which state is the initial one (then it's the id of the
			state), or what are the probability to start in each state (then it's
			a list of probabilities).
		name : str, optional
			Name of the model.
			Default is "unknow_MC"
		"""
		super().__init__(states,initial_state,name)


def HMMtoMC(h) -> MC:
	"""
	Translate a given HMM `h` to a MC

	Parameters
	----------
	h : HMM
		The HMM to translate.

	Returns
	-------
	MC
		A MC equilavent to `h`.
	"""
	states_g = []
	for i,sh in enumerate(h.states):
		transitions = [[],[],[]]
		for sy in range(len(sh.output_matrix[0])):
			for ne in range(len(sh.next_matrix[0])):
				transitions[0].append(sh.output_matrix[0][sy]*sh.next_matrix[0][ne])
				transitions[1].append(sh.next_matrix[1][ne])
				transitions[2].append(sh.output_matrix[1][sy])
		states_g.append(MC_state([],i))
		states_g[-1].transition_matrix = transitions
	return MC(states_g,h.initial_state)


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
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	c = 0
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(MC_state([],c))
		else:
			p = [ float(i) for i in l[:-2].split(' ')]
			l = f.readline()
			s = [ int(i) for i in l[:-2].split(' ') ]
			l = f.readline()
			o = l[:-2].split(' ')
			states.append(MC_state([(s[i],o[i],p[i]) for i in range(len(p))],c))
		c += 1
		l = f.readline()

	return MC(states,initial_state,name)

def MC_random(nb_states: int,alphabet: list,random_initial_state: bool=False) -> MC:
	"""
	Generate a random MC.

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
	A pseudo-randomly generated MC.
	"""
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	
	states = []
	for i in range(nb_states):
		states.append(MC_state([],i))
		states[-1].transition_matrix = [randomProbabilities(len(obs)),s,obs]
	
	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return MC(states,init,"MCGT_random_"+str(nb_states)+"_states")