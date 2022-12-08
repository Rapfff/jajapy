from numpy import ndarray, array, isnan, where
from ast import literal_eval
from copy import deepcopy
class Parametric_Model:
	"""
	Abstract class that represents parametric MC/CTMC/MDP.
	Is inherited by PMC, PCTMC and PMDP.
	Should not be instantiated itself!
	"""
	def __init__(self, matrix: ndarray, labeling: list,
				 parameter_values: list, parameter_indexes: list,
				 parameter_str: list, name: str) -> None:
		"""
		Creates an abstract model for MC, CTMC and MDP.

		Parameters
		----------
		matrix : ndarray
			Represents the transition matrix.
		labeling: list of str
			A list of N observations (with N the nb of states).
			If `labeling[s] == o` then state of ID `s` is labelled by `o`.
			Each state has exactly one label.
		parameter_values: list of float/str
			Contains the value for each parameter.
			`parameter_values[i]` is the instantiation for parameter `i`.
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
			Default is "unknow_MC"
		"""
		self.labeling = labeling
		self.alphabet = list(set(labeling))
		self.nb_states = len(matrix)
		
		if not 'init' in self.labeling:
			msg = "No initial state given: at least one"
			msg += " state should be labelled by 'init'."
			raise ValueError(msg)
		initial_state = [1.0/self.labeling.count("init") if i=='init' else 0.0 for i in self.labeling]


		if len(parameter_indexes) != len(parameter_values) or len(parameter_values) != len(parameter_str):
			raise ValueError("Length of parameter_indexes, parameter_values and parameter_str must be equal.")
		self.nb_parameters = len(parameter_indexes)
		if max(matrix.flatten()) >= self.nb_parameters:
			msg = "Parameter "+str(max(matrix.flatten()))+" found in matrix while length "
			msg+= "of parameter_values is "+str(self.nb_parameters)
			raise ValueError(msg)
		if min(matrix.flatten()) < 0:
			raise ValueError("Parameter "+str(min(matrix.flatten()))+" found in matrix")

		for p in parameter_indexes:
			for i in p:
				if min(i) < 0:
					raise ValueError("State "+str(min(p.flatten()))+" found in parameter_indexes")
				if max(i) >= self.nb_states:
					msg = "State "+str(max(i))+" found in parameter_indexes while there "
					msg+= "are only "+str(self.nb_states)+" states."
					raise ValueError(msg)

		if type(initial_state) == int:
			self.initial_state = array([0.0 for i in range(self.nb_states)])
			self.initial_state[initial_state] = 1.0
		else:
			if round(sum(initial_state)) != 1.0:
				msg = "Error: the sum of initial_state should be 1.0, here it's"
				msg+= str(sum(initial_state))
				raise ValueError(msg)
			self.initial_state = array(initial_state)
		
		if len(self.labeling) != self.nb_states:
			msg = "The length of labeling ("+str(len(labeling))+") is not equal "
			msg+= "to the number of states("+str(self.nb_states)+")"
			raise ValueError(msg)
		
		self.name = name
		self.parameter_values = parameter_values
		self.parameter_indexes= parameter_indexes
		self.parameter_str = parameter_str
		self.matrix = matrix
	
	def isInstantiated(self,state:int = None) -> bool:
		"""
		Checks if all the parameters are instantiated.
		If `state` is set, checks if all the transitions leaving this state
		are instantiated.

		Returns
		-------
		bool
			True if all the parameters are intantiated.
		"""
		if type(state) != type(None):
			return not (isnan(self.parameter_values)==True).any()
		else:
			self._checkStateIndex(state)
			return not (isnan(array([self.parameter_values[i] for i in self.matrix[state]]))==True).any()

	def transitionValue(self,i:int,j:int):
		return self.parameter_values[self.matrix[i,j]]

	def transitionStr(self,i:int,j:int):
		return self.parameter_str[self.matrix[i,j]]

	def instantiate(self,parameters: list, values: list) -> ndarray: 
		"""
		Set all the parameters in `parameters` to the values `values`.

		Parameters
		----------
		parameters : list of string
			List of all parameters to set. This list must contain parameter
			names.
		values : list of float
			List of values. `parameters[i]` will be set to `values[i]`.
		"""
		new_values = deepcopy(self.parameter_values)
		for i,v in zip(parameters,values):
			done = []
			for ind_x,ind_y in self.parameter_indexes[self.parameter_str.index('$'+i+'$')]:
				j = self.matrix[ind_x,ind_y]
				if not j in done:
					done.append(j)
					s = self.parameter_str[j]
					s = s.replace('$'+i+'$',str(v))
					while '$' in s:
						start = s.index('$')
						end = s.index('$',start+1)
						s = s[:start]+ str(self.parameter_values[self.parameter_str.index(s[start+1:end])]) +s[end+1:]
					new_values[j] = eval(s)
		return new_values

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
		return self.labeling[state]
	
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

	def _save(self, f) -> None:
		f.write(self.name)
		f.write('\n')
		f.write(str(self.matrix))
		f.write('\n')
		f.write(str(self.labeling))
		f.write('\n')
		f.write(str(self.parameter_values))
		f.write('\n')
		f.write(str(self.parameter_indexes))
		f.write('\n')
		f.write(str(self.parameter_str))
		f.write('\n')
		f.close()

	def toStormpy(self):
		"""
		Returns the equivalent stormpy sparse model.
		The output object will be a stormpy.SparseDtmc.

		Returns
		-------
		stormpy.SparseDtmc
			The same model in stormpy format.
		"""
		#TODO

	def _checkStateIndex(self,s:int) -> None:
		if type(s) != int:
			raise TypeError('The parameter must be a valid state ID')
		elif s < 0:
			raise IndexError('The parameter must be a valid state ID')
		elif s >= self.nb_states:
			raise IndexError('This model contains only '+str(self.nb_states)+' states')

def loadParametricModel(f):
	name = literal_eval(f.readline()[:-1])
	matrix = literal_eval(f.readline()[:-1])
	labeling = literal_eval(f.readline()[:-1])
	parameter_values = literal_eval(f.readline()[:-1])
	parameter_indexes = literal_eval(f.readline()[:-1])
	parameter_str = literal_eval(f.readline()[:-1])
	return matrix,labeling,parameter_values,parameter_indexes,parameter_str,name