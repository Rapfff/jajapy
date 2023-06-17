from ..base.tools import resolveRandom, randomProbabilities, checkProbabilities
from math import log
from ..base.Base_MC import *
from ..base.Set import Set
from .Scheduler import Scheduler
import numpy.random
from numpy import array, append, dot, zeros, vsplit, reshape, vstack, append, concatenate
from ast import literal_eval
from multiprocessing import cpu_count, Pool

class MDP(Base_MC):
	"""
	Class representing a MDP.
	"""
	def __init__(self,matrix: ndarray, labelling: list, actions:list, name: str="unknown_MDP"):
		"""
		Create a MDP.

		Parameters
		----------
		matrix : ndarray
			Represents the transition matrix.
			`matrix[s1][act_ID][s2][obs_ID]` is the probability of moving 
			from `s1` to `s2` by executing action of ID `act_ID` and seeing 
			the observation of ID `obs_ID`.
		labelling: list of str
			A list of N observations (with N the nb of states).
			If `labelling[s] == o` then state of ID `s` is labelled by `o`.
			Each state has exactly one label.
		actions: list of str
			The list of all possible actions, such that:
			`actions.index("act")` is the ID of `act`.
		name : str, optional
			Name of the model. Default is "unknow_MDP"
		"""
		self.model_type = MDP_ID
		self.actions = actions
		self.nb_actions = len(self.actions)
		super().__init__(matrix,labelling,name)
		for i in range(self.nb_states):
			for a in range(self.nb_actions):
				if not checkProbabilities(matrix[i][a]):
					msg = "The probability to take a transition from state "
					msg+= str(i)+" executing action "+self.actions[a]+" should be 1.0 or 0.0, here it's "+str(matrix[i][a].sum())
					raise ValueError(msg)

	def getActions(self, state:int =-1) -> list:
		"""
		If state is set, returns the list of all the actions available
		in `state`. Otherwise it returns the actions of the model. 


		Parameters
		----------
		state : int, optional
			a state ID

		Returns
		-------
		list of str
			list of actions
		
		Example
		-------
		>>> model.getActions()
		['A','B','C','D']
		>>> model.getActions(1)
		['A','C']
		"""
		if state == -1:
			return self.actions
		else:
			return list(set(self.actions[i] for i in where(self.matrix[state].sum(axis=1) > 0.0)[0]))


	def tau(self,s1: int,action: str,s2: int,obs: str) -> float:
		"""
		Returns the probability of moving from state ``s1`` executing `action`
		to ``s2`` generating observation ``obs``.

		Parameters
		----------
		s1: int
			source state ID.
		action: str
			An action.
		s2: int
			destination state ID.
		obs: str
			generated observation.
		
		Returns
		-------
		float
			A probability.
		
		Example
		-------
		>>> model.tau(0,'A',1,'a')
		0.6
		>>> model.getLabel(0)
		'a'
		>>> model.tau(0,'A',1,'b')
		0.0
		>>> model.tau(0,'B',1,'b')
		0.0
		>>> model.getActions(0)
		['A']		
		"""
		self._checkStateIndex(s1)
		self._checkStateIndex(s2)
		if obs != self.labelling[s1]:
			return 0.0
		if action not in self.actions:
			return 0.0
		return self.matrix[s1][self.actions.index(action)][s2]
	
	def a(self,s1: int,s2: int, action: str) -> float:
		"""
		Returns the probability of moving from state `s1` to state `s2`,
		just after executing `action`.
		Parameters
		----------
		s1 : int
			ID of the source state.		
		s2 : int
			ID of the destination state.
		action: str
			an action.
		
		Returns
		-------
		float
			Probability of moving from state `s1` to state `s2`.
		
		Example
		-------
		>>> model.a(0,1)
		0.6
		"""
		self._checkStateIndex(s1)
		self._checkStateIndex(s2)
		return self.matrix[s1][self.actions.index(action)][s2]
	
	def next(self,state: int, action: str) -> tuple:
		"""
		Return a state-observation pair according to the distributions 
		described by matrix.

		Parameters
		----------
		state: int
			source state ID.
		action: str
			An action.

		Returns
		-------
		output : (int, str)
			A state-observation pair.

		Example
		-------
		>>> model.next(0,'A')
		(1,'a')
		>>> model.getLabel(0)
		'a'
		>>> model.next(0)
		(1,'a')
		>>> model.next(0)
		(2,'a')
		>>> model.a(0,1,'A')
		0.6
		>>> model.a(0,2,'A')
		0.4
		>>> model.next(0,'C')
		(2,'a')
		>>> model.a(0,2,'C')
		1.0
		"""
		c = resolveRandom(self.matrix[state][self.actions.index(action)])
		return (c, self.labelling[state])
			
	def run(self,number_steps: int,scheduler: Scheduler, current: int = -1) -> list:
		"""
		Simulates a run of length ``number_steps`` of the model under
		``scheduler`` and returns the sequence of actions-observations generated.
		
		Parameters
		----------
		number_steps: int
			length of the simulation.
		current : int, optional.
			If current it set, it starts from the state `current`.
			Otherwise it starts from an initial state.

		Returns
		-------
		output: list of str
			List of alterning state-observation.
		"""
		res = []
		if current == -1:
			current = resolveRandom(self.initial_state)
		scheduler.reset()
		current_len = 0
		while current_len < number_steps:
			action = scheduler.getAction()

			while action not in self.getActions(current):
				action = scheduler.getAction()
			
			next_state, observation = self.next(current,action)
			res.append(observation)
			res.append(action)
			scheduler.addObservation(observation)
			current = next_state
			current_len += 1
		res.append(self.labelling[current])
		return res

	def generateSet(self, set_size: int, param, scheduler: Scheduler, distribution=None, min_size=None) -> list:
		"""
		Generates a set (training set / test set) containing `set_size` traces
		generated under ``scheduler``.

		Parameters
		----------
		set_size: int
			number of traces in the output set.
		param: a list, an int or a float.
			the parameter(s) for the distribution. See "distribution".
		scheduler: Scheduler:
			A scheduler used to generated all the traces.
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
		
		Returns
		-------
		output: list
			a set (training set / test set).
		"""
		seq = []
		val = []
		for i in range(set_size):
			if distribution == 'geo':
				curr_size = min_size + int(numpy.random.geometric(param))
			else:
				if type(param) == list:
					curr_size = param[i]
				elif type(param) == int:
					curr_size = param

			trace = self.run(curr_size, scheduler)

			if not trace in seq:
				seq.append(trace)
				val.append(0)

			val[seq.index(trace)] += 1

		return Set(seq,val,t=1)
	
	def save(self,file_path:str) -> None:
		"""Save the model into a text file.

		Parameters
		----------
		file_path : str
			path of the output file.
		
		Examples
		--------
		>>> model.save("my_model.txt")
		"""
		f = open(file_path, 'w')
		f.write("MDP\n")
		f.write(str(self.actions))
		f.write('\n')
		super()._save(f)
	
	def _stateToString(self,state:int) -> str:
		res = "----STATE "+str(state)+"--"+self.labelling[state]+"----\n"
		for ai,a in enumerate(self.actions):
			for s in range(self.nb_states):
				if self.matrix[state][ai][s] > 0.0001:
					res += "s"+str(state)+" - ("+a+") -> s"+str(s)+" : "+str(self.matrix[state][ai][s])+'\n'
		return res
	
	def savePrism(self,file_path:str) -> None:
		"""
		Save this model into `file_path` in the Prism format.

		Parameters
		----------
		file_path : str
			Path of the output file.
		"""
		f = open(file_path,'w')
		f.write("mdp\n\n")
		super()._savePrism(f)
	
	def _savePrismMatrix(self,f) -> None:
		for s1 in range(self.nb_states):
			for ai,a in enumerate(self.actions):
				if (self.matrix[s1,ai]!=0).any():
					res = '\t['+a+'] s='+str(s1)+' ->'
					for s2 in where(self.matrix[s1,ai] != 0.0)[0]:
						res += ' '+str(self.matrix[s1,ai,s2])+":(s'="+str(s2)+") +"
					res = res[:-2]+';\n'
					f.write(res)
		
	def _logLikelihood_oneproc(self,sequences: Set) -> float:
		"""
		Compute the average loglikelihood of a set of sequences.

		Parameters
		----------
		sequences: Set
			set of sequences of actions-observations.
		
		Returns
		-------
		output: float
			loglikelihood of ``sequences`` under this model.
		"""
		sequences_sorted = sequences.sequences[:]
		sequences_sorted.sort()
		loglikelihood = 0.0
		alpha_matrix = self._initAlphaMatrix(len(sequences_sorted[0])//2)
		for seq in range(len(sequences_sorted)):
			sequence_actions = [sequences_sorted[seq][i+1] for i in range(0,len(sequences_sorted[seq])-1,2)]
			sequence_obs = [sequences_sorted[seq][i] for i in range(0,len(sequences_sorted[seq])-1,2)]
			sequence = sequences_sorted[seq]
			times = sequences.times[sequences.sequences.index(sequence)]
			common = 0
			if seq > 0:
				while common < min(len(sequences_sorted[seq-1]),len(sequence)):
					if sequences_sorted[seq-1][common] != sequence[common]:
						break
					common += 1
			common = int(common/2)
			alpha_matrix = self._updateAlphaMatrix(sequence_obs,sequence_actions,common,alpha_matrix)
			
			last_arr = alpha_matrix[-1] * (array(self.labelling) == sequence[-1])
			last_arr = last_arr.sum()
			if last_arr > 0.0:
				loglikelihood += log(last_arr) * times

		return loglikelihood/sum(sequences.times)
	
	def _updateAlphaMatrix(self, sequence_obs: list,
						   sequence_actions:list,
						   common: int, alpha_matrix: list) -> array:
		"""
		Update the given alpha values for all the states for a new
		`sequence_obs` of observations. It keeps the alpha values for the
		``common`` first observations of the ``sequence``. The idea is the 
		following: if you have already computed the alpha values for a previous
		sequence and you want to compute the alpha values of a new sequence
		that starts with the same ``common`` observations you don't need to
		compute again the first ``common`` alpha values for each states. If,
		on the other hand, you have still not computed any alpha values you can
		simply set ``common`` to 0 and give an empty ``alpha_matrix`` which has
		the right size. The method ``initAlphaMatrix`` can generate such matrix.

		Parameters
		----------
		sequence_obs: list of str
			a sequence of observations.
		sequence_actions: list of str
			a sequence of actions.
		common: int
			for each state, the first ``common`` alpha values will be keept
			unchanged.
		alpha_matrix: 2-D narray of float
			the ``alpha_matrix`` to update. Can be generated by the method
			``initAlphaMatrix``.

		Returns
		-------
		output: 2-D narray of float
			the alpha matrix containing all the alpha values for all the states
			for this sequence: ``alpha_matrix[s][t]`` is the probability of
			being in state ``s`` after seing the ``t-1`` first observation of
			``sequence``.
		"""
		diff_size = len(alpha_matrix)-1 - len(sequence_obs)
		if diff_size < 0: # alpha_matrix too small
			n = zeros(-diff_size * self.nb_states).reshape(-diff_size,self.nb_states)
			alpha_matrix = append(alpha_matrix,n,axis=0)
		elif diff_size > 0: #alpha_matrix too big
			alpha_matrix = vsplit(alpha_matrix,[len(alpha_matrix)-diff_size,self.nb_states])[0]
		for k in range(common,len(sequence_obs)):
			for s in range(self.nb_states):
				p = array([self.tau(ss,sequence_actions[k],s,sequence_obs[k]) for ss in range(self.nb_states)])
				alpha_matrix[k+1,s] = dot(alpha_matrix[k],p)
		return alpha_matrix
	
	def _logLikelihood_multiproc(self, sequences: Set) -> float:
		p = Pool(processes = cpu_count()-1)
		tasks = []
		for seq,times in zip(sequences.sequences,sequences.times):
			tasks.append(p.apply_async(self._computeAlphas, [seq, times,]))
		temp = [res.get() for res in tasks if res.get() != False]
		return sum(temp)/sum(sequences.times)
	
	def _computeAlphas(self,sequence: list, times: int) -> float:
		"""
		Compute the alpha values for ``sequence``.

		Parameters
		----------
		sequence: list of str
			Sequence of alternating actions-observations.
		times: int
			Number of times this sequence appears in the sample.

		Returns
		-------
		float
			loglikelihood of ``sequence`` multiplied by ``times``.
		"""
		len_seq = len(sequence)
		prev_arr = array(self.initial_state)
		
		for k in range(0,len_seq-1,2):
			new_arr = zeros(self.nb_states)
			for s in range(self.nb_states):
				p = array([self.tau(ss,sequence[k+1],s,sequence[k]) for ss in range(self.nb_states)])
				new_arr[s] = dot(prev_arr,p)
			prev_arr = new_arr
		prev_arr = prev_arr*(array(self.labelling) == sequence[-1])
		if prev_arr.sum() == 0.0:
			return 0.0
		return log(prev_arr.sum())*times

def loadMDP(file_path: str) -> MDP:
	"""
	Load an MDP saved into a text file.

	Parameters
	----------
	file_path : str
		Location of the text file.
	
	Returns
	-------
	output : MDP
		The MDP saved in `file_path`.
	"""
	f = open(file_path,'r')
	l = f.readline()[:-1] 
	if l != "MDP":
		msg = " this file doesn't describe an MC: it describes a "+l
		raise ValueError(msg)
	actions = literal_eval(f.readline()[:-1])
	labelling = literal_eval(f.readline()[:-1])
	name = f.readline()[:-1]
	initial_state = array(literal_eval(f.readline()[:-1]))
	matrix = literal_eval(f.readline()[:-1])
	matrix = array(matrix)
	f.close()
	return MDP(matrix, labelling, actions, name)


def MDP_random(nb_states: int,labelling: list, actions: list,
	    	   random_initial_state: bool = True, deterministic: bool = False,
			   sseed: int = None) -> MDP:
	"""
	Generate a random MDP.

	Parameters
	----------
	number_states : int
		Number of states.
	labelling : list of str
		List of observations.
	actions : list of str
		List of actions.	
	random_initial_state: bool, optional
		If set to True we will start in each state with a random probability, otherwise we will always start in state 0.
		Default is True.
	deterministic: bool, optional
		If True, the model will be determinstic: in state `s`, with action `a`, for all label `o`,
		there is at most one transition to a state labelled with `o`.
		Default is False.
	sseed : int, optional
		the seed value.
	
	Returns
	-------
	MDP
		A pseudo-randomly generated MDP.
	"""
	if sseed != None:
		seed(sseed)
		numpy.random.seed(sseed)
	alphabet = list(set(labelling))
	matrix = []
	for s in range(nb_states):
		if not deterministic:
			p = array([append(randomProbabilities(nb_states),0.0) for a in actions])
			p = reshape(p,(len(actions),nb_states+1))
			matrix.append(p)
		else:
			matrix.append([])
			for a in actions:
				dest = [choices(where(array(labelling) == o)[0]) for o in alphabet]
				p = zeros(nb_states)
				probs = randomProbabilities(len(alphabet))
				for i,j in zip(dest,probs):
					p[i] = j
				matrix[-1].append(append(p,0.0))

	if random_initial_state:
		init = randomProbabilities(nb_states).tolist()+[0.0]
	else:
		init = [1.0]+[0.0 for i in range(nb_states)]
	matrix.append(array([init for a in actions]))
	matrix = array(matrix)
	
	labelling = labelsForRandomModel(nb_states,labelling)

	seed()
	numpy.random.seed()
	
	return MDP(matrix, labelling, actions, "MDP_random_"+str(nb_states)+"_states")


def createMDP(transitions:list, labelling:list, initial_state, name: str ="unknown_MDP") -> MDP:
	"""
	An user-friendly way to create an MDP.

	Parameters
	----------
	transitions : [ list of tuples (int, str, int, float)]
		Each tuple represents a transition as follow: 
		(source state ID, action, destination state ID, probability).
	labelling: list of str
		A list of N observations (with N the nb of states).
		If `labelling[s] == o` then state of ID `s` is labelled by `o`.
		Each state has exactly one label.
	initial_state : int or list of float
		Determine which state is the initial one (then it's the id of the
		state), or what are the probability to start in each state (then it's
		a list of probabilities).
	name : str, optional
		Name of the model.
		Default is "unknow_MC"
	
	Returns
	-------
	MDP
		the MDP describes by `transitions`, `labelling`, and `initial_state`.
	
	Examples
	--------
	"""
	if 'init' in labelling:
		msg =  "The label 'init' cannot be used: it is reserved for initial states."
		raise SyntaxError(msg)

	states = list(set([i[0] for i in transitions]+[i[2] for i in transitions]))
	states.sort()
	nb_states = len(states)
	actions = list(set([i[1] for i in transitions]))
	actions.sort()
	nb_actions = len(actions)
	
	if nb_states > len(labelling):
		raise ValueError("all states are not labelled (the labelling list is too small).")
	elif nb_states < len(labelling):
		print("WARNING: the labelling list is bigger than the number of states")

	res = zeros((nb_states,nb_actions,nb_states))
	for t in transitions:
		res[states.index(t[0])][actions.index(t[1])][states.index(t[2])] = t[3]
	
	labelling.append('init')
	res = vstack((res,zeros((1,nb_actions,nb_states))))
	res = concatenate((res,zeros((nb_states+1,nb_actions,1))),axis=2)
	if type(initial_state) == int:
		for a in range(nb_actions):
			res[-1][a][initial_state] = 1.0
	else:
		if type(initial_state) == ndarray:
			initial_state = initial_state.tolist()
		for a in range(nb_actions):
			res[-1][a] = array(initial_state+[0.0])
	
	return MDP(res,labelling,actions,name)