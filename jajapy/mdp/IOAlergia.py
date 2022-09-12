from .MDP import *
from math import sqrt, log

class IOFPTA_state:
	"""
	Class for a IOFPTA state
	"""
	def __init__(self, obs: str, c: int, lbl: list) -> None:
		"""
		Creates a IOFTPA state. 

		Parameters
		----------
		obs : str
			The observation associated to this state.
		c : int
			A counter corresponding to the number of times we went through this
			state.
		lbl : list
			A list of alternating action-observation.
			The prefix associated to this state.
		"""
		self.observation = obs
		self.counter = c
		self.label = lbl
		self.id = -1
		self.transitions = {}

	def counterAdd(self,n:int) -> None:
		"""
		In creases the counter by ``n``

		Parameters
		----------
		n : int
			An integer 
		"""
		self.counter += n

	def setId(self,i:int) -> None:
		"""
		Set the ID of this state to ``i``.

		Parameters
		----------
		i : int
			A new state ID.
		"""
		self.id = i

	def successors(self) -> list:
		"""
		Returns the list of all states that can be reached from this state.

		Returns
		-------
		list
			List of state IDs.
		"""
		res = []
		for a in self.transitions:
			for s in self.transitions[a][1]:
				if s not in res:
					res.append(s)
		res.sort()
		return res

	def successorsAction(self,action: str) -> list:
		"""
		Returns the list of all states that can be reached from this state
		executing ``action``.

		Returns
		-------
		list
			List of state IDs.
		"""
		if not self.actionAllowed(action):
			return []
		return self.transitions[action][1]


	def transitionsAdd(self, action: str, prob: float, state: int) -> None:
		"""
		Add a transition leaving this state.

		Parameters
		----------
		action : str
			An action.
		prob : float
			A probability.
		state : int
			A state ID (the destination state).
		"""
		if not self.actionAllowed(action):
			self.transitions[action] = [[prob], [state]]
		else:
			self.transitions[action][0].append(prob)
			self.transitions[action][1].append(state)

	def transitionChange(self, action: str, old_state: int, new_state: int) -> None:
		"""
		Change the destination state of a given transition.

		Parameters
		----------
		action : str
			An action.
		old_state : int
			A state ID (the previous destination).
		new_state : int
			A state ID (the new destination).
		"""
		self.transitions[action][1][self.transitions[action][1].index(old_state)] = new_state

	def getTransitionProb(self,action: str,s2: int) -> float:
		"""
		Returns the probability that we reach ``s2`` from this state executing
		``action``.

		Parameters
		----------
		action : str
			An action.
		s2 : int
			A state ID.

		Returns
		-------
		float
			A probability.
		"""
		return self.transitions[action][0][self.transitions[action][1].index(s2)]

	def actionAllowed(self,act: str) -> bool:
		"""
		Checks if action ``act`` can be executed in this state.

		Parameters
		----------
		act : str
			An action.

		Returns
		-------
		bool
			True if action ``act`` can be executed in this state.
		"""
		return act in self.transitions

	def actionsAllowed(self) -> list:
		"""
		Returns the list of all actions that can be executed in this state.

		Returns
		-------
		list
			A list of actions.
		"""
		return [a for a in self.transitions]

	def __str__(self) -> str:
		return self.label+'\t'+self.transitions



class IOFPTA:
	"""
	Class for a Input and Output Frequency Prefix Tree Acceptor.
	"""
	def __init__(self,states: list, o: list, a: list) -> None:
		"""
		Creates a IOFPTA

		Parameters
		----------
		states : list of IOFPTA_states
			A list of all the states in this IOFPTA
		o : list of str
			A list of all possible observations
		a : list of str
			A list of all possible actions
		"""
		self.states = states
		self.observations = o
		self.actions = a

	def __str__(self) -> str:
		res = ""
		for s in self.states:
			res += str(s)
		return res

	def successor(self, state: int, action: str, obs: str) -> int:
		"""
		Returns the index of the state we reach after executing ``action`` in
		``state`` and seeing ``obs``.

		Parameters
		----------
		state : int
			A state ID.
		action : str
			An action.
		obs : str
			An observation.

		Returns
		-------
		int
			A state ID.
		"""
		if not self.states[state].actionAllowed(action):
			return None

		for i in self.states[state].successorsAction(action):
			if self.states[i].observation == obs:
				return i

	def compatible(self, s1: int, s2: int, epsilon: float) -> bool:
		"""
		Checks if state `s1` and `s2` are ``epsilon`` compatible.

		Parameters
		----------
		s1 : int
			A state ID.
		s2 : int
			A state ID.
		epsilon : float
			A float between 0 and 1 (0 excluded).

		Returns
		-------
		bool
			True if s1 and s2 are epsilon compatible.
		"""
		if s1 == None or s2 == None:
			return True
		
		if self.states[s1].observation != self.states[s2].observation:
			return False

		for a in self.actions:
			for o in self.observations:
				if not self.hoeffding(s1, s2, a, o, epsilon):
					return False
				if not self.compatible(self.successor(s1, a, o), self.successor(s2, a, o), epsilon):
					return False
		return True
				
	def hoeffding(self, s1: int, s2: int, a: str, o: str, epsilon: float) -> bool:
		"""
		Checks if the distance of distributions of state ``s1`` and state
		``s2`` for action ``a`` and observation ``o`` is within the Hoeffding
		bound with parameter ``epsilon``. 

		Parameters
		----------
		s1 : int
			A state ID.
		s2 : int
			A state ID.
		a : str
			An action.
		o : str
			An observation.
		epsilon : float
			A float between 0 and 1 (0 excluded).			

		Returns
		-------
		bool
			True if the distance of distributions are in the Hoeffding bound.
		"""
		i1 = self.successor(s1, a, o)
		i2 = self.successor(s2, a, o)

		if i1 == None or i2 == None:
			return True
				
		f1 = self.states[s1].getTransitionProb(a, i1)
		n1 = sum(self.states[s1].transitions[a][0])
		f2 = self.states[s2].getTransitionProb(a, i2)
		n2 = sum(self.states[s2].transitions[a][0])

		if n1*n2 == 0:
			return True

		return abs((f1/n1)-(f2/n2)) < (sqrt(1/n1)+sqrt(1/n2))*sqrt(log(2/epsilon)/2)

	def findPredec(self,s: int) -> list:
		"""
		Returns a list of pairs state-action (s',a) s.t., by executing action a
		in state s', we can reach state ``s``.

		Parameters
		----------
		s : int
			A state ID.

		Returns
		-------
		list
			A list of pairs stateID-action.
		"""
		res = []
		for i in range(len(self.states)):
			for a in self.states[i].actionsAllowed():
				for j in self.states[i].successorsAction(a):
					if j == s:
						res.append((i,a))
		return res

	def merge(self,s1:int ,s2: int) -> None:
		"""
		Merges state ``s2`` into state ``s1``.

		Parameters
		----------
		s1 : int
			A state ID.
		s2 : int
			A state ID.
		"""
		#action = self.states[s2].label[-2]
		if s1 == None or s2 == None:
			return None

		predec = self.findPredec(s2)
		for sa in predec:
			self.states[sa[0]].transitionChange(sa[1],s2,s1)

		self.states[s1].counterAdd(self.states[s2].counter)
		self.states[s2].counter = 0 #useless but meaningfull

		for a in self.states[s2].actionsAllowed():
			for o in self.observations:
				succ2 = self.successor(s2,a,o)
				if succ2 != None:
					succ1 = self.successor(s1,a,o)
					if succ1 != None:
						self.merge(succ1,succ2)
						#self.states[s1].transition_prob_add(a,succ1,self.states[s2].getTransitionProb(a,succ2))
						#self.states[succ1].counterAdd(self.states[succ2].counter)
						#self.states[succ2].counter = 0 #useless but meaningfull
					else:
						self.states[s1].transitionsAdd(a,self.states[s2].getTransitionProb(a,succ2),succ2)

	def runSeq(self,seq:list) -> bool:
		"""
		Checks if the IOFTPA can generate the given trace ``seq``.

		Parameters
		----------
		seq : list
			A trace, i.e. a list of alternating action-observation.

		Returns
		-------
		bool
			_description_
		"""
		s_current = 0
		i_current = 0
		path = [s_current]
		while i_current < len(seq):
			f = True
			for s in self.states[s_current].successorsAction(seq[i_current]):
				if self.states[s].observation == seq[i_current+1]:
					s_current = s
					path.append(s_current)
					i_current += 2
					f = False
					break
			if f:
				for s in path:
					self.states[s].pprint()
				print()
				print(seq)
				return False
		return True


	def cleanMDP(self,red: list) -> MDP:
		"""
		Returns a MDP built from a subset of states of this IOFTPA.

		Parameters
		----------
		red : list
			A list of state IDs.

		Returns
		-------
		MDP
			An MDP.
		"""
		states = []
		for i in range(len(red)):
			self.states[red[i]].setId(i)

		for i in red:
			new_dic = {}
			dic = self.states[i].transitions
			for a in dic:
				dic[a].append([])
				new_dic[a] = []
				tot = sum(self.states[i].transitions[a][0])
				for j in range(len(dic[a][0])):
					new_dic[a].append((self.states[dic[a][1][j]].id,self.states[dic[a][1][j]].observation,dic[a][0][j]/tot))
			states.append(MDP_state(new_dic, self.observations,len(red),self.actions))
		matrix = array(states)

		return MDP(matrix,self.observations,self.actions,0)



class IOAlergia:
	"""
	Class for an  IOAlergia algorithm.
	This algorithm is described here:
	https://arxiv.org/pdf/1212.3873.pdf
	"""
	def __init__(self):
		None
		
	def _initialize(self,sample,alpha,actions,observations):
		pass
		self.alpha = alpha
		self.sample = sample

		self.actions = actions
		self.observations = observations

		self.actions.sort()
		self.observations.sort()
		self.N = sum(sample.times)
		self.n = len(sample.sequences[0])
		self.t = self._buildIOFPTA()
		self.a = self.t
		
	def _buildIOFPTA(self):
		states_lbl = [[]]
		states = [IOFPTA_state("",self.N,[])]
		#init states_lbl and states_counter
		for i in range(0,self.n,2):
			for seq in range(len(self.sample.sequences)):
				if not self.sample.sequences[seq][:i+2] in states_lbl:
					states_lbl.append(self.sample.sequences[seq][:i+2])
					states.append( IOFPTA_state(self.sample.sequences[seq][i+1], self.sample.times[seq], self.sample.sequences[seq][:i+2]))
				else:
					states[states_lbl.index(self.sample.sequences[seq][:i+2])].counterAdd(self.sample.times[seq])
		#sorting states
		states_lbl.sort()
		for s in states:
			s.setId(states_lbl.index(s.label))
		states_sorted = [None]*len(states)	
		for s in states:
			states_sorted[s.id] = s
		#init states_transitions
		for s1 in range(len(states_sorted)):
			len_s1 = len(states_sorted[s1].label)
			s2 = s1 + 1
			while s2 < len(states_sorted):
				if len(states_sorted[s2].label) < len_s1 + 2: # too short
					s2 += 1
				elif len(states_sorted[s2].label) > len_s1 + 2: # too long
					s2 += 1
				elif states_sorted[s2].label[:-2] != states_sorted[s1].label: # not same prefix
					s2 += 1
				else: # OK
					act = states_sorted[s2].label[-2]
					states_sorted[s1].transitionsAdd(act, states_sorted[s2].counter, s2)
					s2 += 1

		return IOFPTA(states_sorted,self.observations,self.actions)

	def fit(self,sample:Set,epsilon:float) -> MDP:
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		sample : Set
			A trainig set.
		epsilon : float
			Espilon parameter for the compatibility test.
			Should be between 0 and 1 (0 excluded).

		Returns
		-------
		MDP
			Fitted MDP.
		"""
		self.actions, self.observations = sample.getActionsObservations()
		self._initialize(sample,epsilon,self.actions,self.observations)
		red = [0]
		blue = self.a.states[0].successors()

		while len(blue) != 0:
			state_b = blue[0]
			merged = False
			
			for state_r in red:
				if self.t.compatible(state_r,state_b,epsilon):
					self.a.merge(state_r,state_b)
					merged = True
					break

			if not merged:
				red.append(state_b)
			blue.remove(state_b)

			for state_r in red:
				for state_b in self.a.states[state_r].successors():
					if state_b not in red:
						blue.append(state_b)

			blue = list(set(blue))
			blue.sort()

		return self.a.cleanMDP(red)