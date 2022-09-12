from numpy.random import normal
from numpy import array, zeros, ndarray
from ast import literal_eval
from ..base.tools import resolveRandom, randomProbabilities, checkProbabilities
from ..base.Model import Model
from math import sqrt, exp, pi
from random import uniform
	
class GOHMM(Model):
	"""
	Creates a GOHMM.

	Parameters
	----------
	matrix : ndarray
			Represents the transition matrix.
			`matrix[s1][s2]` is the probability of moving from `s1` to `s2`.
	output : ndarray
			Contains the parameters of the guassian distributions.
			`output[s1][0]` is the mu parameter of the distribution in `s1`,
			`output[s1][1]` is the sigma parameter of the distribution in `s1`.
	initial_state : int or list of float
		Determine which state is the initial one (then it's the id of the
		state), or what are the probability to start in each state (then it's
		a list of probabilities).
	name : str, optional
		Name of the model.
		Default is "unknown_GOHMM"
	"""
	def __init__(self,matrix: ndarray,output: ndarray,initial_state,name="unknown_GOHMM"):
		self.output = array(output)
		if min(self.output.T[1]) < 0.0:
			print("ERROR: the sigma parameters must be positive")
			return False
		super().__init__(matrix,initial_state,name)
		for i in range(self.nb_states):
			if not checkProbabilities(matrix[i]):
				print("Error: the probability to take a transition from state",i,"should be 1.0, here it's",matrix[i].sum())
				return False
		

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
		if s1 < 0 or s2 < 0 or s1 >= self.nb_states or s2 >= self.nb_states:
			return 0.0
		return self.matrix[s1][s2]
	
	def mu(self,s: int) -> float:
		"""
		Returns the mu parameter for `s`.

		Parameters
		----------
		s : int
			ID of the source state.

		Returns
		-------
		float
			the mu parameter.
		"""
		return self.output[s][0]

	
	def b(self,s: int, l: float) -> float:
		"""
		Returns the likelihood of generating `l` in state `s`.

		Parameters
		----------
		s : int
			ID of the source state.		
		l : float
			observation.
		
		Returns
		-------
		output : float
			Likelihood of generating `l` in state `s`.
		"""
		mu, sigma  = self.output[s]
		return exp(-0.5*((mu-l)/sigma)**2)/(sigma*sqrt(2*pi))

	def next_obs(self, s:int) -> float:
		"""
		Generates one observation according to the normal distribution in 
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
		mu, sigma  = self.output[s]
		return normal(mu,sigma,1)[0]

	def next_state(self, s:int) -> int:
		"""
		Returns one state ID at random according to the distribution described 
		by the `self.matrix`.
		
		Parameters
		----------
		s : int
			ID of the source state.	
		
		Returns
		-------
		output : int
			A state ID.
		"""
		c = resolveRandom(self.matrix[s].flatten())
		return c

	def next(self, s:int) -> list:
		"""
		Returns a state-observation pair according to the distributions
		described by `self.matrix[s]` and `self.output[s]`.

		Parameters
		----------
		s : int
			ID of the source state.

		Returns
		-------
		output : [int, float]
			A state-observation pair.
		"""
		return [self.next_state(s),self.next_obs(s)]
	
	def tau(self,s1:int,s2:int,obs: float) -> float:
		"""
		Returns the likelihood of generating, from `s1`, observation `obs` 
		while moving to state `s2`.

		Parameters
		----------
		s1 : int
			A state ID.
		s2 : int
			A state ID.
		obs : float
			An observation.

		Returns
		-------
		output : float
			The likelihood of generating, from from `s1`, observation `obs` 
			while moving to state `s2`.
		"""
		return self.a(s1,s2)*self.b(s1,obs)

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
		f.write("GOHMM\n")
		f.write(str(self.output.tolist()))
		f.write('\n')
		super()._save(f)

	def _stateToString(self,state:int) -> str:
		res = "----STATE s"+str(state)+"----\n"
		for j in range(len(self.matrix[state])):
			if self.matrix[state][j] > 0.0001:
					res += "s"+str(state)+" -> s"+str(j)+" : "+str(self.matrix[state][j])+'\n'
		res += "************\n"
		res += "mean: "+str(round(self.output[state][0],4))+'\n'
		res += "std : "+str(round(self.output[state][1],4))+'\n'
		return res


def loadGOHMM(file_path: str) -> GOHMM:
	"""
	Load an GOHMM saved into a text file.

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
	l = f.readline()[:-1] 
	if l != "GOHMM":
		print("ERROR: this file doesn't describe an GOHMM: it describes a "+l)
	output = literal_eval(f.readline()[:-1])
	output = array(output)
	name = f.readline()[:-1]
	initial_state = array(literal_eval(f.readline()[:-1]))
	matrix = literal_eval(f.readline()[:-1])
	matrix = array(matrix)
	f.close()
	return GOHMM(matrix, output, initial_state, name)

def GOHMM_random(nb_states:int,random_initial_state:bool=False,
				 min_mu: float=0.0,max_mu: float=2.0,
				 min_sigma: float=0.5,max_sigma: float=2.0) -> GOHMM:
	"""
	Generates a random GOHMM.

	Parameters
	----------
	nb_states : int
		Number of states.
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

	Returns
	-------
	GOHMM
		A pseudo-randomly generated GOHMM.
	"""
	matrix = []
	output = []
	for s in range(nb_states):
		p1 = array(randomProbabilities(nb_states))
		matrix.append(p1)
		p2 = array([round(uniform(min_mu,max_mu),3),round(uniform(min_sigma,max_sigma),3)])
		output.append(p2)
	matrix = array(matrix)
	output = array(output)

	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return GOHMM(matrix, output, init,"GOHMM_random_"+str(nb_states)+"_states")


def GOHMM_state(transitions:list, output:tuple, nb_states:int) -> ndarray:
	"""
	Given the list of all transition leaving a state `s`, it generates
	the ndarray describing this state `s` in the GOHMM.matrix.
	This method is useful while creating a model manually.

	Parameters
	----------
	transitions : [ list of tuples (int, float)]
		Each tuple represents a transition as follow: 
		(destination state ID, observation, probability).
	output : tuple (float, float)]
		Represents the parameters of the gaussian distribution 
		(mu, sigma).
	alphabet : list
		alphabet of the model in which this state is.
	nb_states: int
		number of states in which this state is

	Returns
	-------
	ndarray
		ndarray describing this state `s` in the GOHMM.matrix.
	"""

	res = zeros(nb_states)
	for t in transitions:
		res[t[0]] = t[1]
	return [res,output]
