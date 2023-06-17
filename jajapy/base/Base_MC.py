from .Model import Model, MC_ID, MDP_ID, CTMC_ID
from numpy import ndarray, where
from random import choices, seed

class Base_MC(Model):
	"""
	Abstract class that represents a model.
	Is inherited by MC, CTMC and MDP.
	Should not be instantiated itself!
	"""
	def __init__(self, matrix: ndarray, labelling: list, name: str) -> None:
		"""
		Creates an abstract model for MC, CTMC and MDP.

		Parameters
		----------
		matrix : ndarray
			Represents the transition matrix.
		labelling: list of str
			A list of N observations (with N the nb of states).
			If `labelling[s] == o` then state of ID `s` is labelled by `o`.
			Each state has exactly one label.
		name : str, optional
			Name of the model.
			Default is "unknow_MC"
		"""
		self.labelling = labelling
		self.alphabet = list(set(labelling))
		
		if not 'init' in self.labelling:
			msg = "No initial state given: at least one"
			msg += " state should be labelled by 'init'."
			raise ValueError(msg)
		initial_state = [1.0/self.labelling.count("init") if i=='init' else 0.0 for i in self.labelling]

		super().__init__(matrix,initial_state,name)
		if len(self.labelling) != self.nb_states:
			raise ValueError("The length of labelling ("+str(len(labelling))+") is not equal to the number of states("+str(self.nb_states)+")")
	
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

	def b(self,state:int, label:str):
		if self.getLabel(state) == label:
			return 1.0
		return 0.0
	
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

	def _savePrism(self,f) -> None:
		f.write("module "+self.name+'\n')
		f.write("\ts: [0.."+str(self.nb_states-1)+"] init "+str(where(self.initial_state==1.0)[0][0])+";\n\n")
		self._savePrismMatrix(f)
		f.write('endmodule\n\n')

		labels = {}
		for s,l in enumerate(self.labelling):
			if l != 'init':
				if not l in labels:
					labels[l] = [str(s)]
				else:
					labels[l].append(str(s))
		for l in labels:
			res = 'label "'+l+'" ='
			for s in labels[l]:
				res += ' s='+s+' |'
			res = res[:-1]+';\n'
			f.write(res)
		f.close()


	def _save(self, f) -> None:
		f.write(str(self.labelling))
		f.write('\n')
		super()._save(f)

	def toStormpy(self):
		"""
		Returns the equivalent stormpy sparse model.
		The output object will be a stormpy.SparseDtmc.

		Returns
		-------
		stormpy.SparseDtmc
			The same model in stormpy format.
		"""
		try:
			from ..with_stormpy import jajapyModeltoStormpy
			return jajapyModeltoStormpy(self)
		except ModuleNotFoundError:
			raise RuntimeError("Stormpy is not installed on this machine.")

def labelsForRandomModel(nb_states: int, labelling: list, sseed:int=None) -> list:
	if 'init' in labelling:
		msg =  "The label 'init' cannot be used: it is reserved for initial states."
		raise SyntaxError(msg)

	if nb_states < len(labelling):
		print("WARNING: the size of the labelling is greater than the",end=" ")
		print("number of states. The last labels will not be assigned to",end=" ")
		print("any states.")
	
	if nb_states > len(labelling):
		print("WARNING: the size of the labelling is lower than the",end=" ")
		print("number of states. The labels for the last states will",end=" ")
		print("be chosen randomly.")
	
	if sseed != None:
		seed(sseed)
	labelling = labelling[:min(len(labelling),nb_states)] + choices(labelling,k=nb_states-len(labelling))
	labelling.append("init")
	seed()
	return labelling
