from ast import literal_eval
from numpy import float64, array, append
class Set:
	"""
	Class representing a set (training set / test set).
	"""
	def __init__(self, sequences:list, times:list = None, from_MDP: bool=False, t: int = None) -> None:
		"""
		Creates the Set.

		Parameters
		----------
		sequences : list
			List of sequences of traces.
			A trace can be:
			- a sequence of observations (list of str) ->HMM or MC
			- an alternating sequence of actions/observations (list of str) ->MDP
			- a sequence of timed observations (list of float) ->GOHMM
			- a sequence of vectors of timed observations (list of list of float) ->MGOHMM
			- an alternating sequence of waiting-times/observations (list of float and str) ->CTMC
		times : list, optional
			`times[i]` is the number of time `sequences[i]` appears in the set.
			Can be omitted.
		from_MDP : bool, optional
			Whether or not the set has been generated by an MDP.
			Default is False.
		t : int, optional
			0 if this set was generated by a HMM or a MC,
			1 if it was generated by a MDP,
			2 if it was generated by a GOHMM,
			3 if it was generated by a MGOHMM,
			4 if it was generated by a CTMC.
		"""
		if sequences == []:
			self.sequences = []
			self.times = []
			self.type = t
			return
		if t != None:
			self.type = t
		else:
			if type(sequences[0][0])  == float64 or type(sequences[0][0])  == float:
				if type(sequences[0][1]) == float64 or type(sequences[0][1])  == float:
					self.type = 2 # GOHMM
				else:
					self.type = 4 # CTMC
			elif from_MDP:
				self.type = 1 # MDP
			elif type(sequences[0][0])  == list:
				self.type = 3 # MGOHMM
			else:
				self.type = 0 # HMM or MC

		if type(times) == type(None):
			self.setFromList(sequences)
		else:
			self.sequences = sequences
			self.times = array(times)
	
	def save(self,file_path: str) -> None:
		"""
		Save this set into a text file.
		
		Parameters
		----------
		file_path: str
			where to save
		"""
		f = open(file_path,'w')
		f.write(str(self.type == 1)+'\n')
		for i in range(len(self.sequences)):
			f.write(str(self.sequences[i])+'\n')
			f.write(str(self.times[i])+'\n')
		f.close()

	def setFromList(self, l: list) -> None:
		"""
		Convert a list of sequences of observations to a set.

		Parameters
		----------
		l : list
			list of sequences of observations.
		"""
		res = [[],[]]
		for s in l:
			s = list(s)
			if s not in res[0]:
				res[0].append(s)
				res[1].append(0)
			res[1][res[0].index(s)] += 1
		self.sequences = res[0]
		self.times = array(res[1])
	
	def isEqual(self, set2) -> bool:
		"""
		Checks wheter or not this set is equal to another set `set2`.

		Parameters
		----------
		set2 : Set
			Another set.

		Returns
		-------
		bool
			True if this set is equal to `set2`.
		"""
		if self.type != set2.type:
			return False
		if sum(self.times) != sum(set2.times):
			return False
		if len(self.sequences) != len(set2.sequences):
			return False
		if len([i for i in self.sequences if i in set2.sequences]) != len(set2.sequences):
			return False
		for s,t1 in zip(self.sequences,self.times):
			try:
				t2 = set2.times[set2.sequences.index(s)]
			except ValueError:
				return False
			if t1 != t2:
				return False
		return True
	
	def getAlphabet(self) -> list:
		"""
		Returns the list of all possible observations in this set.

		Returns
		-------
		list
			list of observations.
		"""
		if self.type == 2 or self.type == 3:
			return [] #doesn't make sense
		observations = []
		if self.type == 4: # timed self.sequences
			for sequence_obs in self.sequences:
				for x in range(1,len(sequence_obs),2):
					if sequence_obs[x] not in observations:
						observations.append(sequence_obs[x])	
		elif self.type == 0:
			for sequence_obs in self.sequences: # non-timed self.sequences
				for x in sequence_obs:
					if x not in observations:
						observations.append(x)
		elif self.type == 1:
			return self.getActionsObservations()[1]

		return observations
	

	def getActionsObservations(self) -> list:
		"""
		Returns all possible observations and all possible actions in this set.

		Returns
		-------
		list
			list of one list of actions and one list of observations.
		"""
		actions = []
		observations = []
		for seq in range(len(self.sequences)):
			sequence_actions = [self.sequences[seq][i] for i in range(0,len(self.sequences[seq]),2)]
			sequence_obs = [self.sequences[seq][i+1] for i in range(0,len(self.sequences[seq]),2)]
			for x in sequence_actions:
				if x not in actions:
					actions.append(x)
			for x in sequence_obs:
				if x not in observations:
					observations.append(x)
		return [actions,observations]
	
	def addSet(self, s2):
		"""
		Merges this set with another generated by the same kind of Markov model.

		Parameters
		----------
		s2 : Set
			set 2.
		
		Raises
		------
		TypeError
			If the two models `type` attribute are different.
		"""
		if self.type != s2.type:
			raise TypeError("The two sets should have been generated by the same"+
							"kind of Markov model.")
		for i in range(len(s2.sequences)):
			if not s2.sequences[i] in self.sequences:
				self.sequences.append(s2.sequences[i])
				self.times = append(self.times, s2.times[i])
			else:
				self.times[self.sequences.index(s2.sequences[i])] += s2.times[i]

def loadSet(file_path: str) -> Set:
	"""
	Load a training/test set saved into a text file.

	Parameters
	----------
	file_path: str
		location of the file.
	
	Returns
	-------
	Set
		a training/test set.
	"""
	res_set = [[],[]]
	f = open(file_path,'r')
	l = f.readline()
	from_MDP = literal_eval(l[:-1])
	l = f.readline()
	while l:
		res_set[0].append(literal_eval(l[:-1]))
		l = f.readline()
		res_set[1].append(int(l[:-1]))
		l = f.readline()
	f.close()
	return Set(res_set[0],res_set[1],from_MDP)
