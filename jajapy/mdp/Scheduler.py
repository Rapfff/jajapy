from ..base.tools import resolveRandom

class Scheduler:
	"""
	Abstract class for general Scheduler
	"""
	def __init__(self) -> None:
		pass
		
	def getAction(self) -> str:
		pass
		
	def reset(self) -> None:
		pass
	
	def addObservation(self,obs) -> None:
		pass

class UniformScheduler(Scheduler):
	"""
	Class for an uniform scheduler. An uniform scheduler will always returns
	each action with an equal probability.
	"""
	def __init__(self, actions: list) -> None:
		"""
		Creates a uniform scheduler for the given actions.

		Parameters
		----------
		actions : list of str
			A list of actions.
		"""
		super().__init__()
		self.actions = [[1/len(actions)]*len(actions),actions]
	
	def getAction(self) -> str:
		"""
		Returns the action chosen by the scheduler

		Returns
		-------
		str
			An action.
		"""
		return self.actions[1][resolveRandom(self.actions[0])]


class MemorylessScheduler(Scheduler):
	"""
	Class for memoryless scheduler.
	"""
	def __init__(self,actions: list) -> None:
		"""
		Creates a memoryless scheduler.

		Parameters
		----------
		actions : list of str
			list of ``n`` actions, with ``n`` the number of states in the model
			``actions[n]`` will always be used in state ``n``.
		"""
		super().__init__()
		self.actions = actions

	def getAction(self,current_state:int) -> str:
		"""
		Returns the action used while in state ``current_state``

		Parameters
		----------
		current_state : int
			ID of the current MDP state.

		Returns
		-------
		str
			Action chosen by the scheduler.
		"""
		return self.actions[current_state]

class FiniteMemoryScheduler:
	"""
	Class for finite memoryless scheduler.
	A finite memory scheduler can be seen as a generative automaton that takes
	on input a sequence of observation and returns a sequence of actions.
	"""
	def __init__(self,action_matrix: dict,transition_matrix: dict) -> None:
		"""
		Creates a finite memory scheduler.

		Parameters
		----------
		action_matrix : dict
			Describes the probability for each action to be chosen given the
			current scheduler state as follows:
			``action_matrix = {scheduler_state_x: [[proba1,proba2,...],[action1,action2,...]],
			scheduler_state_y: [[proba1,proba2,...],[action1,action2,...]],
			...}``

		transition_matrix : dict
			Given the a state ``s``and an observation ``o``, returns to which
			state the scheduler move to if it receives ``o`` while in state `s`
			``transition_matrix = {obs1: [scheduler_state_dest_if_current_state_=_0,scheduler_state_dest_if_current_state_=_1,...],
			obs2: [scheduler_state_dest_if_current_state_=_0,scheduler_state_dest_if_current_state_=_1,...]
			...}``
		"""
		super().__init__()
		self.s = 0
		self.action_matrix = action_matrix
		self.transition_matrix = transition_matrix

	def reset(self) -> None:
		"""
		Reset the model to its initial state. Should be done before generating
		a new trace.
		"""
		self.s = 0

	def getAction(self) -> str:
		"""
		Returns the action chosen by the scheduler

		Returns
		-------
		str
			Action chosen by the scheduler.
		"""
		return self.action_matrix[self.s][1][resolveRandom(self.action_matrix[self.s][0])]

	def addObservation(self,obs:str) -> None:
		"""
		Updates the scheduler state according to the current state and the
		given observation ``obs``.

		Parameters
		----------
		obs : str
			An observation.
		"""
		if obs in self.transition_matrix:
			self.s = self.transition_matrix[obs][self.s]

	def getActions(self,s:int=None) -> list:
		"""
		Returns the list of all the possibles actions (and their probabilities)
		that the scheduler can choose if it is in state ``s``. If ``s`` is not
		provided it considers the current scheduler state.

		Parameters
		----------
		s : int, optional
			A scheduler state ID. By default None

		Returns
		-------
		list
			A list containing a list of probabilities and a list of actions:
			[[proba_action1, proba_action2,...],[action1, action2,...]]
		"""
		if s==None:
			return self.action_matrix[self.s]
		else:
			return self.action_matrix[s]

	def getSequenceStates(self,seq_obs: list) -> list:
		"""
		Given a sequence of observations, returns the sequence of states the
		scheduler goes through.

		Parameters
		----------
		seq_obs : list of str
			A sequence of observations.

		Returns
		-------
		list
			A sequence of scheulder states.
		"""
		self.reset()
		res = [0]
		for o in seq_obs:
			self.add_observation(o)
			res.append(self.s)
		return res

	def getProbability(self,action:str,state:int=None) -> float:
		"""
		Given a scheduler state ID and an action, returns the probability to
		execute this action in this state.

		Parameters
		----------
		action : str
			An action.
		state : int, optional
			A state ID. If not provided it considers the current state.
			By default None

		Returns
		-------
		float
			A probability.
		"""
		if state == None:
			state = self.s
		if not action in self.action_matrix[state][1]:
			return 0
		return self.action_matrix[state][0][self.action_matrix[state][1].index(action)]
