from numpy.random import normal
from ast import literal_eval
from ..base.tools import resolveRandom, randomProbabilities
from ..base.Model import Model, Model_state
from math import sqrt, exp, pi
from random import uniform
class GOHMM_state(Model_state):
	"""
	Class for a GOHMM state
	"""
	def __init__(self,next_matrix: list,output_parameters: list,idd: int):
		"""
		Creates a GOHMM_state

		Parameters
		----------
		output_parameters : list of two floats.
			Parameters of the gaussian distribution in this states: [mu,sigma].
		next_matrix : [ list of float, list of int]
			`[[proba_state1,proba_state2,...],[state1,state2,...]]`. `next_matrix[0][x]` is the probability to move to state `next_matrix[1][x]`.
		idd : int
			State ID.
		"""
		super().__init__(next_matrix,idd)
		self.output_parameters = output_parameters

	def mu(self) -> float:
		"""
		Returns the mu parameter for this state.

		Returns
		-------
		float
			the mu parameter.
		"""
		return self.output_parameters[0]

	def a(self,s: int) -> float:
		"""
		Returns the probability of moving, from this state, to state `s`.

		Parameters
		----------
		s : int
			The destination state ID.

		Returns
		-------
		output : float
			The probability of moving, from this state, to state `s`.
		"""
		if s in self.transition_matrix[1]:
			return self.transition_matrix[0][self.transition_matrix[1].index(s)]
		else:
			return 0.0

	def b(self,l: float) -> float:
		"""
		Returns the likelihood of generating, from this state, observation `l`.

		Parameters
		----------
		l : str
			The observation.

		Returns
		-------
		output : float
			The likelihood of generating, from this state, observation `l`.
		"""
		mu, sigma  = self.output_parameters
		return exp(-0.5*((mu-l)/sigma)**2)/(sigma*sqrt(2*pi))

	def next_obs(self) -> float:
		"""
		Generates one observation according to a normal distribution of
		parameters `self.output_parameters`.
		
		Returns
		-------
		output : str
			An observation.
		"""
		mu, sigma  = self.output_parameters
		return normal(mu,sigma,1)[0]

	def next_state(self) -> int:
		"""
		Returns one state according to the distribution described by the `self.next_matrix`.
		
		Returns
		-------
		output : int
			A state ID.
		"""
		return self.transition_matrix[1][resolveRandom(self.transition_matrix[0])]

	def next(self) -> list:
		"""
		Returns a state-observation pair according to the distributions
		described by `self.next_matrix` and `self.output_parameters`.

		Returns
		-------
		output : [int, float]
			A state-observation pair.
		"""
		return [self.next_state(),self.next_obs()]
	
	def tau(self,state:int ,obs: float) -> float:
		"""
		Returns the likelihood of generating, from this state, observation `obs` while moving to state `s`.

		Parameters
		----------
		s : int
			A state ID.
		obs : float
			An observation.

		Returns
		-------
		output : float
			The likelihood of generating, from this state, observation `obs` and moving to state `s`.
		"""
		return self.a(state)*self.b(obs)

	def __str__(self) -> str:
		res = "----STATE s"+str(self.id)+"----\n"
		for j in range(len(self.transition_matrix[0])):
			if self.transition_matrix[0][j] > 0.000001:
				res += "s"+str(self.id)+" -> s"+str(self.transition_matrix[1][j])+" : "+str(self.transition_matrix[0][j])+'\n'
		res += "************\n"
		res += "mean: "+str(round(self.output_parameters[0],4))+'\n'
		res += "std : "+str(round(self.output_parameters[1],4))+'\n'
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
			res += str(self.output_parameters)
			res += '\n'			
			return res

class GOHMM(Model):
	"""
	Creates a GOHMM.

	Parameters
	----------
	states : list of GOHMM_states
		List of states in this GOHMM.
	initial_state : int or list of float
		Determine which state is the initial one (then it's the id of the
		state), or what are the probability to start in each state (then it's
		a list of probabilities).
	name : str, optional
		Name of the model.
		Default is "unknown_GOHMM"
	"""
	def __init__(self,states,initial_state,name="unknown_GOHMM"):
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
		output : float
			Probability of moving from state `s1` to state `s2`
		"""
		return self.states[s1].a(s2)

	def b(self,s: int, l: str) -> float:
		"""
		Returns the likelihood of generating `l` in state `s`.

		Parameters
		----------
		s : int
			ID of the source state.		
		o : str
			observation.
		
		Returns
		-------
		output : float
			Likelihood of generating `o` in state `s`.
		"""
		return self.states[s].b(l)
	
	def tau(self, s1: int, s2: int, obs: float) -> float:
		"""
		Return the likelihood of generating from state `s1` observation `obs` and moving to state `s2`.

		Parameters
		----------
		s1 : int
			ID of the source state.
		s2 : int
			ID of the destination state.
		obs : float
			An observation.

		Returns
		-------
		output : float
			The likelihood of generating from state `s1` observation `obs` and moving to state `s`.
		"""
		return self.states[s1].tau(s2,obs)
	
	def mu(self,s:int) -> float:
		"""
		Returns the mu parameter for state ``s``.

		Parameters
		----------
		s : int
			State ID

		Returns
		-------
		float
			mu parameter for state ``s``.
		"""
		return self.states[s].mu()


def loadGOHMM(file_path: str) -> GOHMM:
	"""
	Loads an GOHMM saved into a text file.

	Parameters
	----------
	file_path : str
		Location of the text file.
	
	Returns
	-------
	output : GOHMM
		The GOHMM saved in `file_path`.
	"""
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	c = 0
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(GOHMM_state([[],[],[]],c))
		else:
			ps = [ float(i) for i in l[:-2].split(' ')]
			l  = f.readline()[:-2].split(' ')
			s  = [ int(i) for i in l ]
			o  = literal_eval(f.readline()[:-1])
			states.append(GOHMM_state([ps,s],o,c))
		c += 1
		l = f.readline()

	return GOHMM(states,initial_state,name)

def GOHMM_random(nb_states:int,random_initial_state:bool=False,min_mu: float=0.0,max_mu: float=2.0,min_sigma: float=0.5,max_sigma: float=2.0) -> GOHMM:
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
	min_mu : float, optional
		lower bound for mu. By default 0.0
	max_mu : float, optional
		upper bound for mu. By default 2.0
	min_sigma : float, optional
		lower bound for sigma. By default 0.5
	max_sigma : float, optional
		upper bound for sigma. By default 2.0

	Returns
	-------
	GOHMM
		A pseudo-randomly generated GOHMM.
	"""
	s = [i for i in range(nb_states)]
	states = []
	for i in range(nb_states):
		d = [round(uniform(min_mu,max_mu),3),round(uniform(min_sigma,max_sigma),3)]
		states.append(GOHMM_state([randomProbabilities(nb_states),s],d,i))
	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return GOHMM(states,init,"GOHMM_random_"+str(nb_states)+"_states")