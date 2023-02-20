from numpy import ndarray, array, isnan, where, nan
from numpy.random import geometric
from ast import literal_eval
from copy import deepcopy
from ..base.Set import Set
from ..base.Model import Model, PCTMC_ID
from sympy import sympify

class Parametric_Model(Model):
	"""
	Abstract class that represents parametric models.
	Is inherited by PCTMC.
	Should not be instantiated itself!
	"""
	def __init__(self, matrix: ndarray, labelling: list,
				 transition_expr: list, parameter_values: dict,
				 parameter_indexes: list, parameter_str: list,
				 name: str) -> None:
		"""
		Creates an abstract model for PMC or PCTMC.

		Parameters
		----------
		matrix : ndarray
			Represents the transition matrix.
			matrix[i,j] is the index, in `transition_expr`, of the symbolic
			representation of the transition from `i` to `j`.
		labelling: list of str
			A list of N observations (with N the nb of states).
			If `labelling[s] == o` then state of ID `s` is labelled by `o`.
			Each state has exactly one label.
		transition_expr: list of str
			Contains the symbolic representation for each transition.
		parameter_values: list of float
			Contains the value for each parameter.
			`parameter_values[i]` is the instantiation for parameter `i`.
			If the ith parameter is not instantiated, `parameter_values[i] == numpy.nan`.
		parameter_indexes: list of ndarray
			Contains the indexes of each transition using each parameter.
			`parameter_indexes[i] = array([[0,1],[2,1]])` means that parameter `i`
			is used by the transition from state 0 to state 1 and from state 2 to state 1.
		parameter_str: list
			Contains the name of each parameter.
			`parameter_str[i]` is the name of parameter `i`.
			Parameter `i` doesn't have a name if `parameter_str[i]==None`.
		name : str, optional
			Name of the model.
		"""
		self.labelling = labelling
		self.alphabet = list(set(labelling))
		self.nb_states = len(matrix)
		
		if not 'init' in self.labelling:
			msg = "No initial state given: at least one"
			msg += " state should be labelled by 'init'."
			raise ValueError(msg)
		initial_state = [1.0/self.labelling.count("init") if i=='init' else 0.0 for i in self.labelling]

		if len(parameter_indexes) != len(parameter_str):
			raise ValueError("Length of parameter_indexes and parameter_str must be equal.")
		self.nb_parameters = len(parameter_indexes)
		if max(matrix.flatten()) >= len(transition_expr):
			msg = "Transition "+str(max(matrix.flatten()))+" found in matrix while length "
			msg+= "of transition_expr is "+str(len(transition_expr))
			raise ValueError(msg)
		if min(matrix.flatten()) < 0:
			raise ValueError("Transition "+str(min(matrix.flatten()))+" found in matrix")
		
		if len(self.labelling) != self.nb_states:
			msg = "The length of labelling ("+str(len(labelling))+") is not equal "
			msg+= "to the number of states("+str(self.nb_states)+")"
			raise ValueError(msg)
		
		self.transition_expr = transition_expr
		k = list(parameter_values.keys())
		for i in k:
			if isnan(parameter_values[i]):
				del parameter_values[i]
		self.parameter_values = parameter_values
		self.parameter_indexes= parameter_indexes
		self.parameter_str = parameter_str

		super().__init__(matrix,initial_state,name)

	def isInstantiated(self,state1:int = None, state2:int = None, param:str =None) -> bool:
		"""
		If nothing is set, checks if all the parameters are instantiated.
		If `state1` is set, checks if all the transitions leaving this state
		are instantiated.
		If `state1` and `state2` are set, checks if the transitions from
		`state1` to `state2` is instantiated.
		If `param` is set, checks if the parameter `param` is instantiated.

		Parameters
		----------
		state1 : int, optional.
			state ID.
		
		state2 : int, optional.
			state ID.
		
		param : str, optional
			Parameter name.
			
		Returns
		-------
		bool
			True if all the requested entity(ies) is/are instantiated.
			See above.
		"""
		if type(param) == str:
			return param in self.parameter_values
		elif type(state1) == type(None):
			return len(set(self.parameter_str)-set(self.parameter_values.keys())) == 0
		else:
			self._checkStateIndex(state1)
			if type(state2) == type(None):
				parameters = self.involvedParameters(state1)
			else:
				parameters = self.involvedParameters(state1,state2)
			for i in parameters:
				if not self.isInstantiated(param=i):
					return False
			return True

	def transitionValue(self,i:int,j:int):
		"""
		Returns the value of the transition from state `i` to `j`.
		If at least one of the paramater in this transition is not
		instantiated, returns the symbolic representation of the transition.

		Parameters
		----------
		i : int
			source state ID.
		j : int
			destination state ID.

		Returns
		-------
		float or symbolic representation
			The value of the transition from state `i` to `j`.
			If at least one of the paramater in this transition is not
			instantiated, returns the symbolic representation of the
			transition.
		"""
		return self.transition_expr[self.matrix[i,j]].evalf(subs=self.parameter_values)

	def transitionExpression(self,i:int,j:int):
		"""
		Returns the symbolic representation of the transition.

		Parameters
		----------
		i : int
			source state ID.
		j : int
			destination state ID.

		Returns
		-------
		symbolic representation
			The symbolic representation of the transition.
		"""
		return self.transition_expr[self.matrix[i,j]]
	
	def parameterValue(self, p:str)-> float:
		"""
		Returns the value of the parameter `p`.
		If `p` is not instantiated, returns numpy.nan.
		If the model doesn't have any parameter `p`, returns numpy.nan.

		Parameters
		----------
		p : str
			parameter name.

		Returns
		-------
		float
			The value of `p` if `p` is instantiated, numpy.nan otherwise.
			If the model doesn't have any parameter `p`, returns numpy.nan.
		"""
		if not p in self.parameter_values:
			return nan
		return self.parameter_values[p]
	
	def parameterIndexes(self, p:str) -> list:
		"""
		Returns the list of all transitions involving `p`.
		If the model doesn't have any parameter `p`, returns an empty list.

		Parameters
		----------
		p : str
			parameter name.

		Returns
		-------
		list
			The list of all transitions involving `p`.
			If the model doesn't have any parameter `p`, returns an empty list.
		
		Examples
		--------
		>>> m.parameterIndexes('x')
		[[4,0],[2,3]]
		>>> # the transitions from state 4 to 0 and from 2 to 3 involve 'x'.
		>>> m.transitionExpression(4,0)
		3*x
		"""
		if not p in self.parameter_str:
			return []
		return self.parameter_indexes[self.parameter_str.index(p)]

	def instantiate(self,parameters: list, values: list) -> ndarray: 
		"""
		Returns a copy of the `self.parameter_values` dict, where all the
		parameters in `parameters` to the values `values`.
		
		Parameters
		----------
		parameters : list of string
			List of all parameters to set. This list must contain parameter
			names.
		values : list of float
			List of values. `parameters[i]` will be set to `values[i]`.

		Returns
		-------
		numpy.ndarray
			A copy of the `self.parameter_values` dict, where all the
			parameters in `parameters` to the values `values`.
		
		Remark
		------
		The output numpy.ndarray will be used by the inherited class to check
		if this instantiation is valid. If so, the `self.parameter_values` dict
		is updated.
		"""
		new_values = deepcopy(self.parameter_values)
		for s,v in zip(parameters,values):
			new_values[s] = v
		return new_values

	#def evaluateString(self,string:str,parameter_values=None):
	#	if parameter_values == None:
	#		parameter_values = self.parameter_values
	#	return sympify(string).evalf(subs=parameter_values)

	def evaluateTransition(self,i:int,j:int, parameter_values: dict = None):
		"""
		Returns the value of the transition from state `i` to `j`.
		If `parameter_values` is given, the parameters are instantiated by
		`parameter_values`, otherwise by `self.parameter_values`.
		If at least one of the paramater in this transition is not
		instantiated, returns the symbolic representation of the transition.

		Parameters
		----------
		i : int
			source state ID.
		j : int
			destination state ID.
		parameter_values: dict, optional
			dictionary with the instantiation of the parameters.
			The keys are the parameter names (str).

		Returns
		-------
		float or symbolic representation
			The value of the transition from state `i` to `j`.
			If at least one of the paramater this transition is not
			instantiated, returns the symbolic representation of the
			transition.
		"""
		if parameter_values == None:
			parameter_values = self.parameter_values
		v =  self.transitionExpression(i,j).evalf(subs=parameter_values)
		try:
			v = float(v)
		except TypeError:
			pass
		return v

	def involvedParameters(self,i: int,j: int = -1) -> list:
		"""
		Returns the parameters involved in the transition from `i` to `j`.
		If `j` is not set, it returns the parameters involved in all the 
		transitions leaving `i`.

		Parameters
		----------
		i : int
			source state ID.
		j : int, optional
			destination state ID.

		Returns
		-------
		list of str
			list of parameter names.
		"""
		if j == -1:
			j = range(self.nb_states)
		else:
			j =  [j]
		res = set()
		for jj in j:
			res = res.union(self.transitionExpression(i,jj).free_symbols)
		res = list(res)
		res = [i.name for i in res]
		return res

	def getLabel(self,state: int) -> str:
		"""
		Returns the label of `state`.

		Parameters
		----------
		state : int
			a state ID

		Returns
		-------
		str
			a label

		Example
		-------
		>>> model.getLabel(2)
		'Label-of-state-2'
		"""
		self._checkStateIndex(state)
		return self.labelling[state]
	
	def getAlphabet(self) -> list:
		"""
		Returns the alphabet of this model.

		Returns
		-------
		list of str
			The alphabet of this model
		
		Example
		-------
		>>> model.getAlphabet()
		['a','b','c','d','done']
		"""
		return self.alphabet
	
	def __str__(self) -> str:
		res = "Name: "+self.name+'\n'
		for i in self.parameter_values.keys():
			if not isnan(self.parameter_values[i]):
				res += i+' = '+str(round(self.parameter_values[i],5))+'\n'
		res += '\n'
		if 1.0 in self.initial_state:
			res += "Initial state: s"+str(where(self.initial_state==1.0)[0][0])+'\n'
		else:
			res += "Initial state: "
			for i in range(len(self.matrix)):
				if self.initial_state[i] >= 0.001:
					res += 's'+str(i)+': '+str(round(self.initial_state[i],3))+', '
			res = res[:-2]+'\n'
		for i in range(self.nb_states):
			res += self._stateToString(i)+'\n'
		res += '\n'
		return res

	def generateSet(self, set_size: int, param, distribution=None, min_size=None, timed: bool=False) -> Set:
		"""
		Generates a set (training set / test set) containing ``set_size`` traces.

		Parameters
		----------
		set_size: int
			number of traces in the output set.
		param: a list, an int or a float.
			the parameter(s) for the distribution. See "distribution".
		distribution: str, optional
			If ``distribution=='geo'`` then the sequence length will be
			distributed by a geometric law such that the expected length is
			``min_size+(1/param)``.
			If distribution==None param can be an int, in this case all the
			seq will have the same length (``param``), or ``param`` can be a
			list of int.
			Default is None.
		min_size: int, optional
			see "distribution". Default is None.
		timed: bool, optional
			Only for timed model. Generate timed or non-timed traces.
			Default is False.
		
		Returns
		-------
		output: Set
			a set (training set / test set).
		
		Examples
		--------
		>>> set1 = model.generateSet(100,10)
		>>> # set1 contains 100 traces of length 10
		>>> set2 = model.generate(100, 1/4, "geo", min_size=6)
		>>> # set2 contains 100 traces. The length of the traces is distributed following
		>>> # a geometric distribution with parameter 1/4. All the traces contains at
		>>> # least 6 observations, hence the average length of a trace is 6+(1/4)**(-1) = 10.
		"""
		seq = []
		val = []
		for i in range(set_size):
			if distribution == 'geo':
				curr_size = min_size + int(geometric(param))
			else:
				if type(param) == list:
					curr_size = param[i]
				elif type(param) == int:
					curr_size = param

			if timed:
				trace = self.run(curr_size,timed)
			else:
				trace = self.run(curr_size)

			if not trace in seq:
				seq.append(trace)
				val.append(1)
			else:
				val[seq.index(trace)] += 1

		return Set(seq,val)

	def _save(self, f) -> None:
		f.write(self.name)
		f.write('\n')
		f.write(str(self.matrix.tolist()))
		f.write('\n')
		f.write(str(self.labelling))
		f.write('\n')
		f.write(str(self.parameter_values))
		f.write('\n')
		f.write(str(self.parameter_indexes))
		f.write('\n')
		f.write(str(self.parameter_str))
		f.write('\n')
		f.write(str([str(i) for i in self.transition_expr]))
		f.write('\n')
		f.close()

	def _checkStateIndex(self,s:int) -> None:
		try:
			s = int(s)
		except TypeError:
			raise TypeError('The parameter must be a valid state ID')
		if s < 0:
			raise IndexError('The parameter must be a valid state ID')
		elif s >= self.nb_states:
			raise IndexError('This model contains only '+str(self.nb_states)+' states')

def loadParametricModel(f):
	"""
	Used by the inherited classes.
	Should not be used directly!
	"""
	name = f.readline()[:-1]
	matrix = array(literal_eval(f.readline()[:-1]))
	labelling = literal_eval(f.readline()[:-1])
	parameter_values = literal_eval(f.readline()[:-1])
	parameter_indexes = literal_eval(f.readline()[:-1])
	parameter_str = literal_eval(f.readline()[:-1])
	transition_expr = literal_eval(f.readline()[:-1])
	transition_expr = [sympify(i) for i in transition_expr]
	return matrix,labelling,parameter_values,parameter_indexes,parameter_str,transition_expr,name