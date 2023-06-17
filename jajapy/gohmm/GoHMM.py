import numpy.random
import random
from numpy import array, zeros, ndarray
from ast import literal_eval
from ..base.tools import randomProbabilities
from ..base.Base_HMM import Base_HMM, GOHMM_ID
from math import sqrt, exp, pi
	
class GoHMM(Base_HMM):
	"""
	Creates a GoHMM.

	Parameters
	----------
	matrix : ndarray
			Represents the transition matrix.
			`matrix[s1][s2]` is the probability of moving from `s1` to `s2`.
	output : ndarray
			Contains the parameters of the guassian distributions.
			`output[s1][0][0]` is the mu parameter of the first distribution in `s1`,
			`output[s1][0][1]` is the sigma parameter of the first distribution in `s1`.
			`output[s1][1][0]` is the mu parameter of the second distribution in `s1`.
			etc...
	initial_state : int or list of float
		Determine which state is the initial one (then it's the id of the
		state), or what are the probability to start in each state (then it's
		a list of probabilities).
	name : str, optional
		Name of the model.
		Default is "unknown_GoHMM"
	"""
	def __init__(self,matrix,output,initial_state,name="unknown_GoHMM"):
		self.model_type = GOHMM_ID
		self.nb_distributions = len(output[0])
		for i in range(len(output)):
			if len(output[i]) != self.nb_distributions:
				raise ValueError("All state must have as much distributions")

		self.output = array(output)
		try:
			if min(self.output.T[1].flatten()) < 0.0:
				raise ValueError("The sigma parameters must be positive")
		except IndexError:
			raise ValueError("All distribution must have two parameters")
		
		super().__init__(matrix,initial_state,name)
		if len(self.output.flatten()) != self.nb_distributions*self.nb_states*2:
			raise ValueError("All distribution must have two parameters")
	
	def mu(self,s: int) -> ndarray:
		"""
		Returns the mu parameters for this state.

		Parameters
		----------
		s : int
			ID of the source state.

		Returns
		-------
		ndarray
			the mu parameters.
		"""
		return self.output[s].T[0]
			
	def mu_n(self,s: int, n: int) -> float:
		"""
		Returns the mu parameters of the `n`th distribution in `s`.

		Parameters
		----------
		s : int
			ID of the source state.
		n : int
			Index of the distribution.

		Returns
		-------
		float
			the mu parameter of the `n`th distribution in `s`.
		"""
		return self.output[s][n][0]

	
	def b(self,s: int, l: list) -> float:
		"""
		Returns the likelihood of generating `l` in state `s`.

		Parameters
		----------
		s : int
			ID of the source state.		
		l : list of float
			list of observations.
		
		Returns
		-------
		output : float
			Likelihood of generating `l` in state `s`.
		"""
		res = 1.0
		for n,ln in enumerate(l):
			res *= self.b_n(s,n,ln)
		return res

	def b_n(self,s:int,n:int,l:float) -> float:
		"""
		Returns the likelihood of generating, from the nth distribution of `s`,
		observation `l`.

		Parameters
		----------
		s : int
			state ID
		n : int
			Index of the distribution.
		l : str
			The observation.
			
		Returns
		-------
		output : float
			Likelihood of generating, from the `n`th distribution of this
			state, observation `l`.
		"""
		mu, sigma  = self.output[s][n]
		return exp(-0.5*((mu-l)/sigma)**2)/(sigma*sqrt(2*pi))

	def next_obs(self, s:int) -> list:
		"""
		Generates n observations according to the n normal distributions in 
		`s`.

		Parameters
		----------
		s : int
			ID of the source state.	

		Returns
		-------
		output : str
			An observation.
		"""
		return [numpy.random.normal(parameter[0],parameter[1],1)[0] for parameter in self.output[s]]
	
	def tau(self,s1:int,s2:int,obs: list) -> float:
		"""
		Returns the likelihood of generating, from `s1`, observation `obs` 
		while moving to state `s2`.

		Parameters
		----------
		s1 : int
			A state ID.
		s2 : int
			A state ID.
		obs : list of floats
			An observation.

		Returns
		-------
		output : float
			The likelihood of generating, from from `s1`, observation `obs` 
			while moving to state `s2`.
		"""
		return super().tau(s1,s2,obs)

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
		f.write("GoHMM\n")
		super()._save(f)

	def _stateToString(self,state:int) -> str:
		res = super()._stateToString(state)
		for n in range(self.nb_distributions):
			res += "mean "+str(n+1)+": "+str(round(self.output[state][n][0],4))+'\n'
			res += "std "+str(n+1)+": "+str(round(self.output[state][n][1],4))+'\n'
		return res


def loadGoHMM(file_path: str) -> GoHMM:
	"""
	Load an GoHMM saved into a text file.

	Parameters
	----------
	file_path : str
		Location of the text file.
	
	Returns
	-------
	output : GoHMM
		The GoHMM saved in `file_path`.
	"""
	f = open(file_path,'r')
	l = f.readline()[:-1] 
	if l != "GoHMM":
		print("ERROR: this file doesn't describe an GoHMM: it describes a "+l)
	output = literal_eval(f.readline()[:-1])
	output = array(output)
	name = f.readline()[:-1]
	initial_state = array(literal_eval(f.readline()[:-1]))
	matrix = literal_eval(f.readline()[:-1])
	matrix = array(matrix)
	f.close()
	return GoHMM(matrix, output, initial_state, name)

def GoHMM_random(nb_states:int,nb_distributions:int,
				  random_initial_state:bool=False,
				  min_mu: float=0.0,max_mu: float=2.0,
				  min_sigma: float=0.5,max_sigma: float=2.0, sseed: int = None) -> GoHMM:
	"""
	Generates a random GoHMM.

	Parameters
	----------
	nu_states : int
		Number of states.
	nb_distributions : int
		Number of distributions in each state.
	alphabet : list of str
		List of observations.
	random_initial_state: bool, optional
		If set to True we will start in each state with a random 
		probability, otherwise we will always start in state 0.
		Default is False.
	min_mu : float, optional
		lower bound for mu. By default 0.0
	max_mu : float, optional
		upper bound for mu. By default 2.0
	min_sigma : float, optional
		lower bound for sigma. By default 0.5
	max_sigma : float, optional
		upper bound for sigma. By default 2.0
	sseed : int, optional
		the seed value.

	Returns
	-------
	GoHMM
		A pseudo-randomly generated GoHMM.
	"""
	matrix = []
	output = []
	if sseed != None:
		random.seed(sseed)
		numpy.random.seed(sseed)
	for s in range(nb_states):
		p1 = array(randomProbabilities(nb_states))
		matrix.append(p1)
		p2 = [[round(random.uniform(min_mu,max_mu),3),round(random.uniform(min_sigma,max_sigma),3)] for i in range(nb_distributions)]
		p2 = array(p2)
		output.append(p2)
	matrix = array(matrix)
	output = array(output)

	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	numpy.random.seed()
	random.seed()
	return GoHMM(matrix, output, init,"GoHMM_random_"+str(nb_states)+"_states")


def createGoHMM(transitions:list, output:list, initial_state:str,name :str = 'unknown_GoHMM') -> GoHMM:
	"""
	An user-friendly way to create a GoHMM.

	Parameters
	----------
	transitions : [ list of tuples (int, int, float)]
		Each tuple represents a transition as follow: 
		(source state ID, destination state ID, probability).
	output : list of list of tuples (float, float)]
		Represents the parameters of the gaussian distributions 
		`[(mu1, sigma1),(mu2, sigma2),...]`.
		`output[0]` contains the parameters of the distributions in state 0
		`output[0][0]` contains the 2 parameters of the first distribution
		in state 0.
		`output[0][0][0]` is the mu parameter of the first distribution in
		state 0, and `output[0][0][1]` is the sigma parameter of the first
		distribution in state 0.
	initial_state : int or list of float
		Determine which state is the initial one (then it's the id of the
		state), or what are the probability to start in each state (then it's
		a list of probabilities).
	name : str, optional
		Name of the model.
		Default is "unknow_GoHMM".
	
	Returns
	-------
	GoHMM
		the GoHMM describes by `transitions`, `emission`, and `initial_state`.
	
	Examples
	--------
	"""
	states = list(set([i[0] for i in transitions]+[i[1] for i in transitions]))
	states.sort()
	nb_distributions = len(output[0])
	nb_states = len(states)
	matrix = zeros((nb_states,nb_states))
	for t in transitions:
		matrix[states.index(t[0])][states.index(t[1])] = t[2]
	output = array((nb_states,nb_distributions,2))
	for s,ts in enumerate(output):
		for i,t in enumerate(ts):
			output[s,i][0] = t[0]
			output[s,i][1] = t[1]
	return GoHMM(matrix, output,initial_state,name)