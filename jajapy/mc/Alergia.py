from .MC import *
from math import sqrt, log
from ..base.Set import Set
from ..base.tools import normalize
class Alergia:
	"""
	class for general ALERGIA algorithm on MC.
	This algorithm is described here:
	https://grfia.dlsi.ua.es/repositori/grfia/pubs/57/icgi1994.pdf
	"""

	def __init__(self):
		None

	def _initialize(self,traces:Set,alpha:float,alphabet:list=None) -> None:
		"""
		Create the PTA from the training set ``traces ``, initialize
		the alpha value and the alphabet.

		Parameters
		----------
		traces : Set
			Training set.
		alpha : float
			alpha value used in the Hoeffding boung
		alphabet : list, optional.
			List of all possible observations, by default None.
		"""
		self.alpha = alpha
		if alphabet == None:
			self.alphabet = traces.getAlphabet()
		else:
			self.alphabet = alphabet
		self.createPTA(traces)
	
	def createPTA(self,traces: Set) -> MC:
		"""
		Create a PTA from ``traces``.

		Parameters
		----------
		traces : Set
			traces used to generate the PTA.
		"""
		N = sum(traces.times)
		n = len(traces.sequences[0])

		self.states_lbl = [""]
		self.states_counter= [N]
		
		#states_transitions = [
		#						[state1: [proba1,proba2,...],[state1,state2,...],[obs1,obs2,...]]
		#						[state2: [proba1,proba2,...],[state1,state2,...],[obs1,obs2,...]]
		#						...
		#					  ]

		self.states_transitions = []

		#init self.states_lbl and self.states_counter
		for i in range(n):
			for seq in range(len(traces.sequences)):
				if not traces.sequences[seq][:i+1] in self.states_lbl:
					self.states_lbl.append(traces.sequences[seq][:1+i])
					self.states_counter.append(traces.times[seq])
				else:
					self.states_counter[self.states_lbl.index(traces.sequences[seq][:i+1])] += traces.times[seq]

		#init self.states_transitions
		for s1 in range(len(self.states_lbl)):
			self.states_transitions.append([[],[],[]])
			
			len_s1 = len(self.states_lbl[s1])
			
			s2 = s1 + 1
			while s2 < len(self.states_lbl):
				if len(self.states_lbl[s2]) == len_s1: # too short
					s2 += 1
				elif len(self.states_lbl[s2]) == len_s1 + 2: # too long
					break
				elif self.states_lbl[s2][:-1] != self.states_lbl[s1]: # not same prefix
					s2 += 1
				else: # OK
					self.states_transitions[-1][0].append(self.states_counter[s2])
					self.states_transitions[-1][1].append(s2)
					self.states_transitions[-1][2].append(self.states_lbl[s2][-1])
					s2 += 1

	def fit(self,traces: Set,alpha: float=0.1,alphabet: list=None) -> MC:
		"""
		Fits a MC according to ``traces``.

		Parameters
		----------
		traces : Set
			The training set.
		alpha : float, optional
			_description_, by default 0.1
		alphabet : list, optional
			The alphabet of the model we are learning.
			Can be omitted.
			
		Returns
		-------
		MC
			fitted MC.
		"""
		self._initialize(traces,alpha,alphabet)
		
		for j in range(1,len(self.states_lbl)):
			if self.states_lbl[j] != None:
				for i in range(j):
					if self.states_lbl[i] != None:
						if self._compatibleMerge(i,j):
							j -= 1
							break

		return self._toMC()

	def _transitionStateAction(self,state,action):
		try:
			return self.states_transitions[state][1][self.states_transitions[state][2].index(action)]
		except ValueError:
			return None
	
	def _areDifferent(self,i:int,j:int,a:str) -> bool:
		"""
		return if nodes ``i`` and ``j`` are different for the observation
		``a`` according to the Hoeffding bound computed  with ``self.alpha``.

		Parameters
		----------
		i : int
			index of the first node.
		j : int
			index of the second node.
		a : str
			observation.

		Returns
		-------
		bool
			``True`` if the are different, ``False`` otherwise.
		"""
		ni = self.states_counter[i]
		nj = self.states_counter[j]
		try:
			fi = self.states_transitions[i][0][self.states_transitions[i][2].index(a)]
		except ValueError:
			fi = 0
		try:
			fj = self.states_transitions[j][0][self.states_transitions[j][2].index(a)]
		except ValueError:
			fj = 0
		return ( abs((fi/ni) - (fj/nj)) > sqrt(0.5*log(2/self.alpha))*((1/sqrt(ni)) + (1/sqrt(nj))) )


	def _compatibleMerge(self,i,j):
		choices = [0]
		pairs = [(i,j)]

		while True:
			a = self.alphabet[choices[-1]]
			if self._areDifferent(i,j,a): #stop
				return False

			i = self._transitionStateAction(i,a)
			j = self._transitionStateAction(j,a)


			if i == None or j == None or i == j:
				i = pairs[-1][0]
				j = pairs[-1][1]
				choices[-1] += 1 #next 
				
				while choices[-1] == len(self.alphabet):#roll back
					choices = choices[:-1]
					self._merge(i,j)
					pairs = pairs[:-1]
					if len(pairs) == 0:
						return True
					i = pairs[-1][0]
					j = pairs[-1][1]
					choices[-1] += 1

			else: #deeper
				choices.append(0)
				pairs.append((i,j))

	def _merge(self,i,j):
		if i > j :
			j,i = i,j
		for state in range(len(self.states_lbl)):
			if self.states_lbl[state] != None:
				for transition in range(len(self.states_transitions[state][1])):
					if self.states_transitions[state][1][transition] == j:
						self.states_transitions[state][1][transition] = i
		
		for a in self.states_transitions[j][2]:
			
			ja = self.states_transitions[j][2].index(a)
			
			if a in self.states_transitions[i][2]:
				self.states_transitions[i][0][self.states_transitions[i][2].index(a)] += self.states_transitions[j][0][ja]
			
			else:
				self.states_transitions[i][0].append(self.states_transitions[j][0][ja])
				self.states_transitions[i][1].append(self.states_transitions[j][1][ja])
				self.states_transitions[i][2].append(a)


		self.states_counter[i] += self.states_counter[j]
		self.states_lbl[j] = None
		self.states_transitions[j] = None
		self.states_counter[j] = None

	def _toMC(self):
		self.nb_states = len(self.states_lbl) - self.states_lbl.count(None)
		states = []
		c = -1 
		for i in range(len(self.states_transitions)):
			if self.states_lbl[i] != None:
				c+=1
				self.states_transitions[i][0] = [j/self.states_counter[i] for j in self.states_transitions[i][0]]
				self.states_transitions[i][0] = normalize(self.states_transitions[i][0])
				self.states_transitions[i][1] = [j-self.states_lbl[:j].count(None) for j in self.states_transitions[i][1]]
				l = list(zip(self.states_transitions[i][1], self.states_transitions[i][2], self.states_transitions[i][0]))
				l = MC_state(l, self.alphabet, self.nb_states)
				states.append(l)
		states = array(states)
		return MC(states, self.alphabet,0)
