from numpy.random import normal
from ast import literal_eval
from ..base.tools import randomProbabilities
from ..gohmm.GOHMM import *
from math import sqrt, exp, pi
from random import uniform
class MGOHMM_state(GOHMM_state):
	"""
	Class for a MGOHMM state
	"""
	def __init__(self,next_matrix: list,output_parameters: list,idd: int):
		"""
		Creates a MGOHMM_state

		Parameters
		----------
		output_parameters : list of lists of two floats.
			Parameters of the gaussian distribution in this states: [[mu_1,sigma_1],[mu_2,sigma_2],...].
		next_matrix : [ list of tuples (int, float)]
			Each tuple represents a transition as follow: 
			(destination state ID, probability).
		idd : int
			State ID.
		"""
		super().__init__(next_matrix, output_parameters,idd)

	def mu(self) -> list:
		"""
		Returns the mu parameters for this state.

		Returns
		-------
		list
			the mu parameters.
		"""
		return [i[0] for i in self.output_parameters]
		
	def mu_n(self,n:int) -> float:
		"""
		Returns the mu parameter for the nth distribution in this state.

		Parameters
		----------
		n : int
			Index of the distribution.

		Returns
		-------
		float
			the nth mu parameter.
		"""
		return self.output_parameters[n][0]

	def b(self,l: list) -> float:
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
		res = 1.0
		for n,ln in enumerate(l):
			res *= self.b_n(n,ln)
		return res
	
	def b_n(self,n:int,l:float) -> float:
		"""
		Returns the likelihood of generating, from the nth distribution of this
		state, observation `l`.

		Parameters
		----------
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
		mu, sigma  = self.output_parameters[n]
		return exp(-0.5*((mu-l)/sigma)**2)/(sigma*sqrt(2*pi))

	def next_obs(self) -> list:
		"""
		Generates n observations according to the n normal distributions of
		parameters `self.output_parameters`.
		
		Returns
		-------
		output : list of float
			An list of n observations.
		"""
		return [normal(parameter[0],parameter[1],1)[0] for parameter in self.output_parameters]

	def next(self) -> list:
		"""
		Returns a state-observation pair according to the distributions
		described by `self.next_matrix` and `self.output_parameters`.

		Returns
		-------
		output : [int, list of floats]
			A state-observation pair.
		"""
		return [self.next_state(),self.next_obs()]
	
	def tau(self,state:int ,obs: list) -> float:
		"""
		Returns the likelihood of generating, from this state, observation
		`obs` while moving to state `s`.

		Parameters
		----------
		s : int
			A state ID.
		obs : list
			An observation.

		Returns
		-------
		output : float
			The likelihood of generating, from this state, observation `obs`
			and moving to state `s`.
		"""
		return self.a(state)*self.b(obs)
	
	def __str__(self) -> str:
		res = "----STATE s"+str(self.id)+"----\n"
		for j in range(len(self.transition_matrix[0])):
			if self.transition_matrix[0][j] > 0.000001:
				res += "s"+str(self.id)+" -> s"+str(self.transition_matrix[1][j])+" : "+str(self.transition_matrix[0][j])+'\n'
		res += "************\n"
		for n in range(len(self.output_parameters)):
			res += "mean "+str(n+1)+": "+str(round(self.output_parameters[n][0],4))+'\n'
			res += "std "+str(n+1)+": "+str(round(self.output_parameters[n][1],4))+'\n'
		return res


class MGOHMM(GOHMM):
	"""
	Creates a MGOHMM.

	Parameters
	----------
	states : list of MGOHMM_states
		List of states in this MGOHMM.
	initial_state : int or list of float
		Determine which state is the initial one (then it's the id of the
		state), or what are the probability to start in each state (then it's
		a list of probabilities).
	name : str, optional
		Name of the model.
		Default is "unknown_MGOHMM"
	"""
	def __init__(self,states,initial_state,name="unknown_MGOHMM"):
		if min([len(i.output_parameters) for i in states]) != max([len(i.output_parameters) for i in states]):
			print("Error")
			return False
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

	def b_n(self,s: int, n: int, l: float) -> float:
		"""
		Returns the likelihood of generating, from the nth distribution of 
		state `n`, observation `l`.

		Parameters
		----------
		s : int
			ID of the source state.	
		n : int
			Index of the distribution.	
		l : float
			observation.
		
		Returns
		-------
		output : float
			Likelihood of generating `l` with the `n`th distribution of state
			`s`.
		"""
		return self.states[s].b_n(n,l)

	def b(self,s: int, l: list) -> float:
		"""
		Returns the likelihood of generating `l` in state `s`.

		Parameters
		----------
		s : int
			ID of the source state.		
		l : list of float
			A list of observations.
		
		Returns
		-------
		output : float
			Likelihood of generating `l` in state `s`.
		"""
		return self.states[s].b(l)
	
	def tau(self, s1: int, s2: int, obs: list) -> float:
		"""
		Return the likelihood of generating from state `s1` observation `obs` and moving to state `s2`.

		Parameters
		----------
		s1 : int
			ID of the source state.
		s2 : int
			ID of the destination state.
		obs : list of float
			A list of observations.

		Returns
		-------
		output : float
			The likelihood of generating from state `s1` observation `obs` and moving to state `s`.
		"""
		return self.states[s1].tau(s2,obs)
	
	def mu(self,s:int) -> list:
		"""
		Returns the mu parameters for state ``s``.

		Parameters
		----------
		s : int
			State ID

		Returns
		-------
		list of floats
			mu parameters for state ``s``.
		"""
		return self.states[s].mu()
	
	def mu_n(self,s:int,n:int) -> float:
		"""
		Returns the mu parameter for the nth distribution in state `s`.

		Parameters
		----------
		s : int
			State ID
		n : int
			Index of the distribution.

		Returns
		-------
		float
			The nth mu parameter.
		"""
		return self.state[s].mu_n(n)


def loadMGOHMM(file_path: str) -> MGOHMM:
	"""
	Loads an MGOHMM saved into a text file.

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
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	c = 0
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(MGOHMM_state([],[],c))
		else:
			ps = [ float(i) for i in l[:-2].split(' ')]
			l  = f.readline()[:-2].split(' ')
			s  = [ int(i) for i in l ]
			o  = literal_eval(f.readline()[:-1])
			states.append(MGOHMM_state(list(zip(s,ps)),o,c))
		c += 1
		l = f.readline()

	return MGOHMM(states,initial_state,name)

def MGOHMM_random(nb_states:int,nb_distributions:int,
				  random_initial_state:bool=False,
				  min_mu: float=0.0, max_mu: float=2.0,
				  min_sigma: float=0.5,max_sigma: float=2.0) -> MGOHMM:
	"""
	Generates a random HMM.

	Parameters
	----------
	nb_states : int
		Number of states.
	nb_distributions : int
		Number of distributions in each state.
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
	MGOHMM
		A pseudo-randomly generated MGOHMM.
	"""
	s = [i for i in range(nb_states)]
	states = []
	for i in range(nb_states):
		d = []
		for _ in range(nb_distributions):
			d.append([round(uniform(min_mu,max_mu),3),round(uniform(min_sigma,max_sigma),3)])
		states.append(MGOHMM_state(list(zip(s,randomProbabilities(nb_states))),d,i))
	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return MGOHMM(states,init,"MGOHMM_random_"+str(nb_states)+"_states")