from random import random
from scipy.stats import norm
from numpy import nditer, array, ndarray


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
	
	Examples
	--------
	>>> import jajapy as ja
	>>> p = [0.2,0.6,0.15,0.05]
	>>> chosen = ja.resolveRandom(p)
	>>> print(chosen)
	1
	"""
	while True:
		r = random()
		i = 0
		while r > sum(m[:i+1]) and i < len(m):
			i += 1
		if i < len(m):
			break
	return i

def normalize(ll):
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
		return array([i/ll.sum() for i in nditer(ll)])

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
	
	Raises
	------
	ValueError
		If `size` is strictly lower than 1.0.
	TypeError
		If `size` is not an `int`.
	
	Examples
	--------
	>>> import jajapy as ja
	>>> p = ja.randomProbabilities(4)
	>>> print(p)
	[0.3155861636575178, 0.11453783121165262, 0.5686125794652406, 0.001263425665589013]
	"""
	if size <= 0:
		raise ValueError("The size parameter should be higher than 0.")
	if type(size) != int:
		raise TypeError("The size parameter should be an int.")
	rand = []
	for i in range(size-1):
		rand.append(random())
	rand.sort()
	rand.insert(0,0.0)
	rand.append(1.0)
	return [rand[i]-rand[i-1] for i in range(1,len(rand))]

def checkProbabilities(l: ndarray) -> bool:
	"""
	Check if a ndarray contains probabilities i.e. check if the sum of the
	ndarray is 0.0 or 1.0. 

	Parameters
	----------
	l : ndarray
		List of probabilities

	Returns
	-------
	bool
		True if the sum of the list is 0.0 or 1.0.
	"""
	return round(l.sum(),3) == 1.0