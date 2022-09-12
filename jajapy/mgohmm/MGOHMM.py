from numpy.random import normal
from numpy import array, zeros, ndarray
from ast import literal_eval
from ..base.tools import resolveRandom, randomProbabilities, checkProbabilities
from ..base.Model import Model
from math import sqrt, exp, pi
from random import uniform
	
class MGOHMM(Model):
	"""
	Creates a MGOHMM.

	Parameters
	----------
	matrix : ndarray
			Represents the transition matrix.
			`matrix[s1][s2]` is the probability of moving from `s1` to `s2`.
	output : ndarray of  or list
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
		Default is "unknown_MGOHMM"
	"""
	def __init__(self,matrix,output,initial_state,name="unknown_MGOHMM"):
		self.nb_distributions = len(output[0])
		for i in range(len(output)):
			if len(output[i]) != self.nb_distributions:
				print("ERROR: all state must have as much distributions")
				return False
		self.output = array(output)
		if min(self.output.T[1].flatten()) < 0.0:
			print("ERROR: the sigma parameters must be positive")
			return False
		super().__init__(matrix,initial_state,name)
		for i in range(self.nb_states):
			if not checkProbabilities(matrix[i]):
				print("Error: the probability to take a transition from state",i,"should be 1.0, here it's",matrix[i].sum())
				return False

		if len(self.output.flatten()) != self.nb_distributions*self.nb_states*2:
			print("ERROR: all distribution must have two parameters")
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
		return [normal(parameter[0],parameter[1],1)[0] for parameter in self.output[s]]

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
		output : [int, list of floats]
			A state-observation pair.
		"""
		return [self.next_state(s),self.next_obs(s)]
	
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
		f.write("MGOHMM\n")
		f.write(str(self.output.tolist()))
		f.write('\n')
		super()._save(f)

	def _stateToString(self,state:int) -> str:
		res = "----STATE s"+str(state)+"----\n"
		for j in range(len(self.matrix[state])):
			if self.matrix[state][j] > 0.0001:
					res += "s"+str(state)+" -> s"+str(j)+" : "+str(self.matrix[state][j])+'\n'
		res += "************\n"
		for n in range(self.nb_distributions):
			res += "mean "+str(n+1)+": "+str(round(self.output[state][n][0],4))+'\n'
			res += "std "+str(n+1)+": "+str(round(self.output[state][n][1],4))+'\n'
		return res


def loadMGOHMM(file_path: str) -> MGOHMM:
	"""
	Load an MGOHMM saved into a text file.

	Parameters
	----------
	file_path : str
		Location of the text file.
	
	Returns
	-------
	output : MGOHMM
		The MGOHMM saved in `file_path`.
	"""
	f = open(file_path,'r')
	l = f.readline()[:-1] 
	if l != "MGOHMM":
		print("ERROR: this file doesn't describe an MGOHMM: it describes a "+l)
	output = literal_eval(f.readline()[:-1])
	output = array(output)
	name = f.readline()[:-1]
	initial_state = array(literal_eval(f.readline()[:-1]))
	matrix = literal_eval(f.readline()[:-1])
	matrix = array(matrix)
	f.close()
	return MGOHMM(matrix, output, initial_state, name)

def MGOHMM_random(nb_states:int,nb_distributions:int,
				  random_initial_state:bool=False,
				  min_mu: float=0.0,max_mu: float=2.0,
				  min_sigma: float=0.5,max_sigma: float=2.0) -> MGOHMM:
	"""
	Generates a random MGOHMM.

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

	Returns
	-------
	MGOHMM
		A pseudo-randomly generated MGOHMM.
	"""
	matrix = []
	output = []
	for s in range(nb_states):
		p1 = array(randomProbabilities(nb_states))
		matrix.append(p1)
		p2 = [[round(uniform(min_mu,max_mu),3),round(uniform(min_sigma,max_sigma),3)] for i in range(nb_distributions)]
		p2 = array(p2)
		output.append(p2)
	matrix = array(matrix)
	output = array(output)

	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return MGOHMM(matrix, output, init,"MGOHMM_random_"+str(nb_states)+"_states")


def MGOHMM_state(transitions:list, output:list, nb_states:int) -> ndarray:
	"""
	Given the list of all transition leaving a state `s`, it generates
	the ndarray describing this state `s` in the MGOHMM.matrix.
	This method is useful while creating a model manually.

	Parameters
	----------
	transitions : [ list of tuples (int, float)]
		Each tuple represents a transition as follow: 
		(destination state ID, observation, probability).
	output : list of tuples (float, float)]
		Represents the parameters of the gaussian distributions 
		[(mu1, sigma1),(mu2, sigma2),...].
	alphabet : list
		alphabet of the model in which this state is.
	nb_states: int
		number of states in which this state is

	Returns
	-------
	ndarray
		ndarray describing this state `s` in the MGOHMM.matrix.
	"""
	res = zeros(nb_states)
	for t in transitions:
		res[t[0]] = t[1]
	return [res,output]
