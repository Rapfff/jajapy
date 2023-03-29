from .MC import MC
from ..base.Set import Set
from ..base.tools import hoeffdingBound
from numpy import zeros

class PTA_node:
	def __init__(self, label,parent=None,count=0) -> None:
		self.label = label
		self.count = count
		self.kids = []
		self.parent = parent
	
	def findKid(self,label):
		for (k,_) in self.kids:
			if k.label == label:
				return k
		return None

	def incKid(self,el,c=1):
		if type(el) == str:
			label = el
			s = -1
		else:
			s = el
			label = -1
		for i,(k,_) in enumerate(self.kids):
			if k.label == label or s==k:
				self.kids[i] = (k,self.kids[i][1]+c)
				return k
		return None

	def replaceKid(self,s1,s2):
		for i,(k,_) in enumerate(self.kids):
			if k == s1:
				self.kids[i] = (s2,self.kids[i][1])
				return True
		return None
	
class PTA:
	def __init__(self,traces) -> None:
		self.root = PTA_node('')
		self.alphabet = []
		
		for trace,time in zip(traces.sequences,traces.times):
			node = self.root
			curr = 0
			while curr < len(trace):
				char = trace[curr]
				if not char in self.alphabet:
					self.alphabet.append(char)
				nnext = node.incKid(char,time)
				if nnext == None:
					node = self.addKids(node,trace[curr:],time)
					curr = len(trace)
				else:
					node = nnext
					curr += 1
			node.count = time

	def addKids(self,start:PTA_node,string:str,count:int):
		n = start
		for l in string:
			n.kids.append((PTA_node(l,n),count))
			n = n.kids[-1][0]
		return n

	def compatible(self,s1,s2,alpha):
		if s1 == None or s2 == None:
			return True
		if s1.label != s2.label:
			return False
		n1 = sum([i for (_,i) in s1.kids])+s1.count
		n2 = sum([i for (_,i) in s2.kids])+s2.count
		if not hoeffdingBound(s1.count,n1,s2.count,n2,alpha):
			return False
		for char in self.alphabet:
			k1,k2 = s1.findKid(char),s2.findKid(char)
			if not self.compatible(k1,k2,alpha):
				return False
		return True
		
	def merge(self,s1,s2):
		s1.count += s2.count
		s2.parent.replaceKid(s2,s1)
		self.fold(s1,s2)

	def fold(self,s1,s2):
		for (k2,c) in s2.kids:
			k1 = s1.findKid(k2.label)
			if k1 == None:
				s1.kids.append((k2,c))
				k2.parent = s1
			else:
				k1.count += k2.count
				s1.incKid(k1,c)
				self.fold(k1,k2)

	def pprint(self):
		print(self.toMC())

	def toMC(self):
		root = self.root
		if 'init' in self.alphabet:
			if len(self.root.kids) == 1 and self.root.kids[0][0].label == 'init':
				root = self.root.kids[0][0]
			else:
				msg =  "The label 'init' cannot be used: it is reserved for initial states."
				raise SyntaxError(msg)
		else:
			root.label = 'init'
		
		states = [root]
		temp = [root]
		while len(temp) != 0:
			n = temp[0]
			temp = temp[1:]
			for (k,c) in n.kids:
				if not k in states:
					states.append(k)
					temp.append(k)
		
		matrix = zeros((len(states),len(states)))
		for i,s in enumerate(states):
			for (k,c) in s.kids:
				matrix[i,states.index(k)] = c
			if (matrix[i]>0).any():
				matrix[i] /= matrix[i].sum()
		labelling = [s.label for s in states]
		return MC(matrix,labelling)
		

class Alergia:
	"""
	Class for general ALERGIA algorithm described here:
	https://grfia.dlsi.ua.es/repositori/grfia/pubs/57/icgi1994.pdf
	"""

	def __init__(self):
		None

	def fit(self,traces: Set,alpha: float=0.1,
			stormpy_output: bool = True, output_file_prism: str = None):
		"""
		Fits a MC according to ``traces``.

		Parameters
		----------
		traces : Set
			The training set.
		alpha : float, optional
			_description_, by default 0.1
		stormpy_output: bool, optional
			If set to True the output model will be a Stormpy sparse model.
			Default is True.
		output_file_prism : str, optional
			If set, the output model will be saved in a prism file at this
			location. Otherwise the output model will not be saved.

		Returns
		-------
		MC or stormpy.SparseDtmc
			The fitted MC.
			If `stormpy_output` is set to `False` or if stormpy is not available on
			the machine it returns a `jajapy.MC`, otherwise it returns a `stormpy.SparseDtmc`
		"""
		self._initialize(traces,alpha)
		red = [self.T.root]
		blue = [k for (k,_) in self.T.root.kids]
		while len(blue)>0:
			s1 = blue[0]
			merged = False
			for s2 in red:
				if self.T.compatible(s1,s2,alpha):
					self.T.merge(s2,s1)
					merged = True
					break
			if not merged:
				red.append(s1)
			blue = []
			for s in red:
				blue += [k for (k,_) in s.kids]
			blue = list(set(blue))
			blue = [s for s in blue if s not in red]
		m = self.T.toMC()

		try:
			from ..with_stormpy import jajapyModeltoStormpy
			stormpy_installed = True
		except ModuleNotFoundError:
			stormpy_installed = False
		if stormpy_output and not stormpy_installed:
			print("WARNING: stormpy not found. The output model will not be a stormpy sparse model")
			stormpy_output = False
		
		if output_file_prism:
			self.h.savePrism(output_file_prism)

		if stormpy_output:
			return jajapyModeltoStormpy(m)
		else:
			return m

	def _initialize(self,traces:Set,alpha:float) -> None:
		"""
		Create the PTA from the training set ``traces ``, initialize
		the alpha value and the alphabet.

		Parameters
		----------
		traces : Set
			Training set.
		alpha : float
			alpha value used in the Hoeffding bound
		"""
		self.alpha = alpha
		self.alphabet = traces.getAlphabet()
		self.createPTA(traces)

	
	def createPTA(self,traces):
		temp1 = traces.sequences[:]
		temp2 = traces.times[:]
		traces.sequences.sort()
		traces.times = [temp2[temp1.index(i)] for i in traces.sequences]
		self.T = PTA(traces)
		
		
