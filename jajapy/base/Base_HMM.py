from .tools import resolveRandom, checkProbabilities
from .Model import Model, HMM_ID, GOHMM_ID

class Base_HMM(Model):
	"""
	Abstract class that represents a model.
	Is inherited by HMM and GoHMM.
	Should not be instantiated itself!
	"""

	def __init__(self,matrix, initial_state, name):
		"""
		Creates an abstract model for HMM and GoHMM.

		Parameters
		----------
		matrix : ndarray
			Represents the transition matrix.
			`matrix[s1][s2]` is the probability of moving from `s1` to `s2`.
		initial_state : int or list of float
			Determine which state is the initial one (then it's the id of the
			state), or what are the probability to start in each state (then it's
			a list of probabilities).
		name : str, optional
			Name of the model.
		"""
		super().__init__(matrix,initial_state,name)
		for i in range(self.nb_states):
			if not checkProbabilities(matrix[i]):
				raise ValueError("The probability to take a transition from state",i,"should be 1.0, here it's",matrix[i].sum())
				

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
		self._checkStateIndex(s1)
		self._checkStateIndex(s2)
		return self.matrix[s1][s2]

	def tau(self, s1: int, s2: int, obs) -> float:
		"""
		Return the probability of generating from state `s1` observation `obs` and moving to state `s2`.
		If `s1` or `s2` is not a valid state ID it returns 0.

		Parameters
		----------
		s1 : int
			ID of the source state.
		s2 : int
			ID of the destination state.
		obs : str or list
			An observation. `str` if the model is an HMM,`list` if it's a GoHMM

		Returns
		-------
		float
			The probability of generating from state `s1` observation `obs` and moving to state `s`.
		"""
		return self.a(s1,s2)*self.b(s1,obs)

	def _stateToString(self, state: int) -> str:
		res = "----STATE s"+str(state)+"----\n"
		for j in range(len(self.matrix[state])):
			if self.matrix[state][j] > 0.0001:
				res += "s"+str(state)+" -> s"+str(j)+" : "+str(self.matrix[state][j])+'\n'
		res += "************\n"
		return res
	
	def _save(self,f) -> None:
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
		Returns one state ID at random according to the distribution described 
		by the `self.matrix`.
		
		Parameters
		----------
		state : int
			ID of the source state.	
		
		Returns
		-------
		int
			A state ID.
		"""
		c = resolveRandom(self.matrix[state].flatten())
		return c

	def next(self, state: int) -> list:
		"""
		Returns a state-observation pair according to the distributions
		described by `self.matrix[s]` and `self.output[s]`.

		Parameters
		----------
		state : int
			ID of the source state.

		Returns
		-------
		output : [int, list of floats]
			A state-observation pair.
		"""
		return [self.next_state(state),self.next_obs(state)]

