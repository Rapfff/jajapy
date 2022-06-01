from random import random
from scipy.stats import norm
from numpy import nditer
from ast import literal_eval


def normpdf(x: float, params: list, variation:float = 0.01) -> float:
	"""
	Returns the probability of ``x`` under a normal distribution returns of 
	parameters ``params``. Since this probability should be 0 it returns in
	fact the probability that the normal distribution gives us a value
	between ``x-variation`` and ``x+variation``.

	Parameters
	----------
	x: float
		the value of the normal distribution.
	params: list of two float.
		the parameters of the distribution: [mean,sd].
	variation: float
		the vicinity.

	Returns
	-------
	float
		the probability of ``x`` under a normal distribution returns of
		parameters ``params``.
	"""
	return norm.cdf(x+variation,params[0],params[1]) - norm.cdf(x-variation,params[0],params[1])


def loadSet(file_path:str , float_obs: bool = False) -> list:
	"""
	Load a training/test set saved into a text file.

	Parameters
	----------
	file_path: str
		location of the file.
	float_obs: bool, optional.
		Should be True if the observations are float. By default is Flase.
	
	Returns
	-------
	float
		a training/test set.
	"""
	res_set = [[],[]]
	f = open(file_path,'r')
	l = f.readline()
	while l:
		res_set[0].append(literal_eval(l[:-1]))
		l = f.readline()
		res_set[1].append(int(l[:-1]))
		l = f.readline()
	f.close()
	return res_set

def saveSet(t_set: list, file_path: str) -> None:
	"""
	Save a training/test set into a text file.
	
	Parameters
	----------
	t_set: list
		the training/test set to save
	file_path: str
		where to save
	"""
	f = open(file_path,'w')
	for i in range(len(t_set[0])):
		f.write(str(t_set[0][i])+'\n')
		f.write(str(t_set[1][i])+'\n')
	f.close()

def resolveRandom(m: list) -> int:
	"""
	Given a list of probabilities it returns the index of the one choosen
	according to the probabilities.
	Example: if m=[0.7,0.3], it will returns 0 with probability 0.7,
	and 1 with probability 0.3.
	
	Parameters
	----------
	m: list of float
		list of probabilities.
	
	Returns
	-------
	int
		the chosen index.
	"""
	while True:
		r = random()
		i = 0
		while r > sum(m[:i+1]) and i < len(m):
			i += 1
		if i < len(m):
			break
	return i

def correct_proba(ll):
	"""
	Normalizes a list of float such that the sum is equal to 1.0.

	Parameters
	----------
	ll: list of float or 1-D narray
		list of probabilities to normalize.

	Returns
	-------
	list or 1-D narray
		normalized list.
	"""
	if type(ll) == list:
		return [i/sum(ll) for i in ll]
	else:
		return [i/ll.sum() for i in nditer(ll)]

def randomProbabilities(size: int) -> list:
	"""
	Return of list l of length ``size`` of probabilities.

	Parameters
	----------
	size: int
		size of the output list.

	Returns
	-------
	list of float
		list of probabilities.
	"""
	rand = []
	for i in range(size-1):
		rand.append(random())
	rand.sort()
	rand.insert(0,0.0)
	rand.append(1.0)
	return [rand[i]-rand[i-1] for i in range(1,len(rand))]

def mergeSets(s1: list, s2: list) -> list:
	"""
	Merges two sets (training set / test set).

	Parameters
	----------
	s1 : list
		set 1.
	s2 : list
		set 2.

	Returns
	-------
	list
		two sets merged.
	"""
	for i in range(len(s2[0])):
		if not s2[0][i] in s1[0]:
			s1[0].append(s2[0][i])
			s1[1].append(s2[1][i])
		else:
			s1[1][s1[0].index(s2[0][i])] += s2[1][i]
	return s1


def getAlphabetFromSequences(sequences: list) -> list:
	"""
	Returns the list of all possible observations in a list of sequences of 
	observations.

	Parameters
	----------
	sequences : list
		list of sequences of observations.

	Returns
	-------
	list
		list of observations.
	"""
	sequences = sequences[0]
	observations = []
	if type(sequences[0][0]) == float: # timed sequences
		for sequence_obs in sequences:
			for x in range(1,len(sequence_obs),2):
				if sequence_obs[x] not in observations:
					observations.append(sequence_obs[x])	
	else:
		for sequence_obs in sequences: # non-timed sequences
			for x in sequence_obs:
				if x not in observations:
					observations.append(x)
	return observations

def getActionsObservationsFromSequences(sequences: list ) -> list:
	"""
	Returns all possible observations and all possible actions in a list of sequences of 
	observations.

	Parameters
	----------
	sequences : list
		list of sequences of observations.

	Returns
	-------
	list
		list of one list of actions and one list of observations.
	"""
	sequences = sequences[0]
	actions = []
	observations = []
	for seq in range(len(sequences)):
		sequence_actions = [sequences[seq][i] for i in range(0,len(sequences[seq]),2)]
		sequence_obs = [sequences[seq][i+1] for i in range(0,len(sequences[seq]),2)]
		for x in sequence_actions:
			if x not in actions:
				actions.append(x)
		for x in sequence_obs:
			if x not in observations:
				observations.append(x)

	return [actions,observations]

def setFromList(l: list) -> list:
	"""
	Convert a list of sequences of observations to a set.

	Parameters
	----------
	l : list
		list of sequences of observations.

	Returns
	-------
	list
		a set (training set / test set)
	"""
	res = [[],[]]
	for s in l:
		s = list(s)
		if s not in res[0]:
			res[0].append(s)
			res[1].append(0)
		res[1][res[0].index(s)] += 1
	return res