from ..base.tools import resolveRandom, randomProbabilities, checkProbabilities
from math import log
from ..base.Model import Model
from ..base.Set import Set
from .Scheduler import Scheduler
from numpy.random import geometric
from numpy import array, append, dot, zeros, vsplit, ndarray, where, reshape
from ast import literal_eval
from multiprocessing import cpu_count, Pool

class MDP(Model):
	"""
	Class representing a MDP.
	"""
	def __init__(self,matrix: ndarray, alphabet: list,
				 actions: list,initial_state,name: str="unknown_MDP"):
		"""
		Create a MDP.

		Parameters
		----------
		matrix : ndarray
			Represents the transition matrix.
			`matrix[s1][act_ID][s2][obs_ID]` is the probability of moving 
			from `s1` to `s2` by executing action of ID `act_ID` and seeing 
			the observation of ID `obs_ID`.
		alphabet: list
			The list of all possible observations, such that:
			`alphabet.index("obs")` is the ID of `obs`.
		alphabet: list
			The list of all possible observations, such that:
			`alphabet.index("obs")` is the ID of `obs`.
		initial_state : int or list of float
			Determine which state is the initial one (then it's the id of the
			state), or what are the probability to start in each state (then it's
			a list of probabilities).
		name : str, optional
			Name of the model. Default is "unknow_MDP"
		"""
		self.actions = actions
		self.nb_actions = len(actions)
		self.alphabet = alphabet
		super().__init__(matrix,initial_state,name)
		for i in range(self.nb_states):
			for a in range(self.nb_actions):
				if not checkProbabilities(matrix[i][a]) and round(matrix[i][a].sum(),3) != 0.0:
					print("Error: the probability to take a transition from",end=" ")
					print("state",i,"executing",actions[a],"should be",end=" ")
					print("1.0 or 0.0, here it's",matrix[i][a].sum())
					return False

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
		"""
		if state == -1:
			return self.actions
		else:
			return [self.actions[i] for i in where(self.matrix[state].sum(axis=2) > 0.0)[0]]

	def getAlphabet(self,state: int = -1) -> list:
		"""
		If state is set, returns the list of all the observations we could
		see in `state`. Otherwise it returns the alphabet of the model. 

		Parameters
		----------
		state : int, optional
			a state ID

		Returns
		-------
		list of str
			list of observations
		"""
		if state == -1:
			return self.alphabet
		else:
			return [self.alphabet[i] for i in where(self.matrix[state].sum(axis=0).sum(axis=0) > 0.0)[0]]

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
		"""
		if s1 < 0 or s1 > self.nb_states or not action in self.actions or not obs in self.alphabet:
			return 0.0
		return self.matrix[s1][self.actions.index(action)][s2][self.alphabet.index(obs)]
	
	def next(self,state: int, action: str) -> tuple:
		"""
		Return a state-observation pair according to the distributions 
		described by matrix

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
		"""
		c = resolveRandom(self.matrix[state][self.actions.index(action)].flatten())
		return (c//len(self.alphabet), self.alphabet[c%len(self.alphabet)])
			
	def run(self,number_steps: int,scheduler: Scheduler) -> list:
		"""
		Simulates a run of length ``number_steps`` of the model under
		``scheduler`` and returns the sequence of actions-observations generated.
		
		Parameters
		----------
		number_steps: int
			length of the simulation.

		Returns
		-------
		output: list of str
			List of alterning state-observation.
		"""
		res = []
		current = resolveRandom(self.initial_state)
		scheduler.reset()
		current_len = 0
		while current_len < number_steps:
			action = scheduler.getAction()

			while action not in self.getActions(current):
				action = scheduler.getAction()
			
			res.append(action)
			next_state, observation = self.next(current,action)
			res.append(observation)
			scheduler.addObservation(observation)
			
			current = next_state
			current_len += 1
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
				curr_size = min_size + int(geometric(param))
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

		return Set(seq,val,from_MDP=True)
	
	def _stateToString(self,state:int) -> str:
		res = "----STATE s"+str(state)+"----\n"
		for ai,a in enumerate(self.actions):
			for s in range(self.nb_states):
				for oi,o in enumerate(self.alphabet):
					if self.matrix[state][ai][s][oi] > 0.0001:
						res += "s"+str(state)+" - ("+a+") -> s"+str(s)+" : "+o+" : "+str(self.matrix[state][ai][s][oi])+'\n'
		return res
	
	def save(self,file_path:str):
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
		f.write(str(self.alphabet))
		f.write('\n')
		f.write(str(self.actions))
		f.write('\n')
		super()._save(f)

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
			sequence_actions = [sequences_sorted[seq][i] for i in range(0,len(sequences_sorted[seq]),2)]
			sequence_obs = [sequences_sorted[seq][i+1] for i in range(0,len(sequences_sorted[seq]),2)]
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
			if alpha_matrix[-1].sum() > 0:
				loglikelihood += log(alpha_matrix[-1].sum()) * times

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
		for k in range(0,len_seq,2):
			new_arr = zeros(self.nb_states)
			for s in range(self.nb_states):
				p = array([self.tau(ss,sequence[k],s,sequence[k+1]) for ss in range(self.nb_states)])
				new_arr[s] = dot(prev_arr,p)
			prev_arr = new_arr
		if prev_arr.sum() == 0.0:
			return 0.0
		return log(prev_arr.sum())*times

	#-------------------------------------------

	def saveToPrism(self,output_file:str) -> None:
		"""
		Save the MDP into a file with the Prism format.
		WARNING: This works only if we start in a given state
		with probability 1.0.

		Parameters
		----------
		output_file : str 
			Where to save the output Prism MDP.
		"""
		if not 1.0 in self.initial_state:
			print("ERROR: in PRISM the initial state is deterministic (not stochastic).")
			return None
		f = open(output_file,'w',encoding="utf-8")
		f.write("mdp\n")
		f.write("\tmodule "+self.name+"\n")
		f.write("\ts : [0.."+str(self.nb_states-1)+"] init "+str(self.initial_state.index(1.0))+";\n")
		f.write("\tl : [0.."+str(len(self.observations()))+"] init "+str(len(self.observations()))+";\n")
		f.write("\n")
		for s in range(self.nb_states):
			for a in self.getActions(s):
				ai = self.actions.index(a)
				f.write("\t["+a+"] s="+str(s)+" -> ")
				res = ""
				for s2 in range(self.nb_states):
					for oi,o in enumerate(self.alphabet):
						if self.matrix[s][a][s2][oi] > 0.0:
							res += self.matrix[s][a][s2][oi]+" : (s'="+str(s2)+") & (l'="+o+") + "
				res = res[:-3]
				res+= ";\n"
				f.write(res)
		f.write("\n")
		f.write("endmodule\n")

		for l in range(len(self.alphabet)):
			f.write('label "'+self.alphabet[l]+'" = l='+str(l)+';\n')
		f.close()

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
		print("ERROR: this file doesn't describe an MDP: it describes a "+l)
	alphabet = literal_eval(f.readline()[:-1])
	actions = literal_eval(f.readline()[:-1])
	name = f.readline()[:-1]
	initial_state = array(literal_eval(f.readline()[:-1]))
	matrix = literal_eval(f.readline()[:-1])
	matrix = array(matrix)
	f.close()
	return MDP(matrix, alphabet, actions, initial_state, name)


def MDP_random(nb_states: int,alphabet: list,actions: list,random_initial_state: bool = False) -> MDP:
	"""
	Generate a random MDP.

	Parameters
	----------
	number_states : int
		Number of states.
	alphabet : list of str
		List of observations.
	actions : list of str
		List of actions.	
	random_initial_state: bool, optional
		If set to True we will start in each state with a random probability, otherwise we will always start in state 0.
		Default is False.
	
	Returns
	-------
	MDP
		A pseudo-randomly generated MDP.
	"""
	matrix = []
	for s in range(nb_states):
		matrix.append([])
		for a in actions:
			p = array(randomProbabilities(nb_states*len(alphabet)))
			p = reshape(p, (nb_states,len(alphabet)))
			matrix[-1].append(p)
	matrix = array(matrix)

	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return MDP(matrix, alphabet, actions, init,"MDP_random_"+str(nb_states)+"_states")


def MDPFileToPrism(file_path:str,output_file:str) -> None:
	"""
	Translate a MDP save file into a MDP Prism save file.
	
	Parameters
	----------
	file_path : str
		The MDP save file.
	output_file : str 
		Where to save the output Prism MDP.
	"""
	m = loadMDP(file_path)
	m.saveToPrism(output_file)

def loadPrismMDP(file_path:str) -> MDP:
	"""
	Load an MDP saved into a text file with the Prism format.

	Parameters
	----------
	file_path : str
		Location of the Prism file.
	
	Returns
	-------
	MDP
		The MDP saved in `file_path`.
	"""
	f = open(file_path)
	f.readline()
	f.readline()
	l = f.readline()
	l = l.split(' ')

	states = []
	init = int(l[-1][:-2])
	for i in range(int(l[2][4:-1])+1):
		states.append({})

	actions = []
	alphabet = []

	l = f.readline()
	while l[:-1] != "endmodule":
		act = l[1]
		actions.append(act)
		state = int(l[l.find('=')+1:l.find('-')-1])
		l = (' '+f.readline()).split('+')
		states[state][act] = []
		states[state][act].append([ float(i[1:i.find(':')-1]) for i in l ]) #add proba
		states[state][act].append([ int(i[i.find('=')+1:i.find(')')]) for i in l ]) #add state

		l = f.readline()
	
	actions = list(set(actions))

	map_s_o = {}
	l = f.readline()

	while l:
		l = l[:-2]
		if not "goal" in l:
			obs = l[l.find('"')+1:l.rfind('"')]
			obs = obs[0].upper() + obs[1:]
			alphabet.append(obs)
			l = l.split('|')
			s = [int(i[i.rfind('=')+1:]) for i in l]
			for ss in s:
				map_s_o[ss] = obs
		l = f.readline()

	alphabet = list(set(alphabet))

	for state in range(len(states)):
		for a in states[state]:
			o = [ map_s_o[states[state][a][1][i]] for i in range(len(states[state][a][1])) ]
			p = states[a][0]
			s = states[a][1]
			states[state][a] = list(zip(s,o,p))


	states = [MDP_state(j,i) for i,j in enumerate(states)]

	m = MDP(array(states),alphabet,actions,init,file_path[:-6])
	return m

def MDP_state(transitions:dict, alphabet:list, nb_states:int, actions: list) -> ndarray:
	"""
	Given the list of all transition leaving a state `s`, it generates
	the ndarray describing this state `s` in the MDP.matrix.
	This method is useful while creating a model manually.

	Parameters
	----------
	transition : dict
			transition = {action1 : [(destination_state_1,observation_1,proba_1),(destination_state_1,observation_1,proba_1)...],
			action2 : [(destination_state_1,observation_1,proba_1),(destination_state_1,observation_1,proba_1)...],
			...}
	alphabet : list
		alphabet of the model in which this state is.
	nb_states: int
		number of states in which this state is
	actions : list
		actions of the model in which this state is.

	Returns
	-------
	ndarray
		ndarray describing this state `s` in the MDP.matrix.
	"""

	res = zeros((len(actions),nb_states,len(alphabet)))
	for a in transitions:
		for t in transitions[a]:
			res[actions.index(a)][t[0]][alphabet.index(t[1])] = t[2]
	return res
