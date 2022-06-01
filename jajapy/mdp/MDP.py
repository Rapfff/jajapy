from ..base.tools import resolveRandom, randomProbabilities
from math import log
from ..base.Model import Model, Model_state
from .Scheduler import Scheduler
from numpy.random import geometric
from numpy import array, append, dot, zeros, vsplit
from ast import literal_eval

class MDP_state(Model_state):
	"""
	Class for a MDP state
	"""

	def __init__(self,transition_matrix: dict, idd : int) -> None:
		"""
		Creates a MDP state

		Parameters
		----------
		transition_matrix : dict
			transition_matrix = {action1 : [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_obs,transition2_obs,...]],
			action2 : [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_obs,transition2_obs,...]]
			...}
		idd : int
			State ID
		"""
		super().__init__(transition_matrix, idd)

	def next(self,action: str) -> list:
		"""
		Return a state-observation pair according to the distributions
		described by transition_matrix and the given executed action.

		Parameters
		----------
		action: str
			An action.

		Returns
		-------
		output : [int, str]
			A state-observation pair.
		"""
		if not action in self.transition_matrix:
			print("Error: action",action,"is not available in state",self.idd)
			return False
		c = resolveRandom(self.transition_matrix[action][0])
		return [self.transition_matrix[action][1][c],self.transition_matrix[action][2][c]]
	
	def tau(self,action: str,state: int,obs: str) -> float:
		"""
		Returns the probability of generating, from this state and using
		`action`, observation `obs` while moving to state `state`.

		Parameters
		----------
		action: str
			An action.
		state : int
			A state ID.
		obs : str
			An observation.

		Returns
		-------
		output : float
			A probability.
		"""
		if action not in self.actions():
			return 0.0
		for i in range(len(self.transition_matrix[action][0])):
			if self.transition_matrix[action][1][i] == state and self.transition_matrix[action][2][i] == obs:
				return self.transition_matrix[action][0][i]
		return 0.0
	
	def observations(self) -> list:
		"""
		Return the list of all the observations that can be generated from this state.

		Returns
		-------
		output : list of str
			A list of observations.
		"""
		obs = []
		for a in self.actions():
			obs += [o for o in self.transition_matrix[a][2]]
		return list(set(obs))

	def actions(self) -> list:
		"""
		Return the list of all the actions that can be executed in this state.

		Returns
		-------
		output : list of str
			A list of actions.
		"""
		return [i for i in self.transition_matrix]

	def __str__(self) -> str:
		res = "----STATE s"+str(self.id)+"----\n"
		for action in self.transition_matrix:
			for j in range(len(self.transition_matrix[action][0])):
				if self.transition_matrix[action][0][j] > 0.0001:
					res += "s"+str(self.id)+" - ("+action+") -> s"+str(self.transition_matrix[action][1][j])+" : "+self.transition_matrix[action][2][j]+' : '+str(self.transition_matrix[action][0][j])+'\n'
		return res

	def save(self) -> str:
		if len(self.transition_matrix) == 0: #end state
			return "-\n"
		res = ""
		for action in self.transition_matrix:
			res += str(action)
			res += '\n'
			for proba in self.transition_matrix[action][0]:
				res += str(proba)+' '
			res += '\n'
			for state in self.transition_matrix[action][1]:
				res += str(state)+' '
			res += '\n'
			for obs in self.transition_matrix[action][2]:
				res += str(obs)+' '
			res += '\n'
		res += "*\n"
		return res

class MDP(Model):
	"""
	Class representing a MDP.
	"""
	def __init__(self,states: list,initial_state,name: str="unknown_MDP"):
		"""
		Create a MDP.

		Parameters
		----------
		states : list of MDP_states
			List of states in this MDP.
		initial_state : int or list of float
			Determine which state is the initial one (then it's the id of the
			state), or what are the probability to start in each state (then it's
			a list of probabilities).
		name : str, optional
			Name of the model. Default is "unknow_MDP"
		"""
		super().__init__(states,initial_state,name)

	def actions(self) -> list:
		"""
		Returns all the actions for this model. Warning: some actions may be
		unavailable in some states.

		Returns
		-------
		list of str
			A list of actions.
		"""
		res = []
		for s in self.states:
			res += s.actions()
		res = list(set(res))
		res.sort()
		return res

	def actionsState(self,s:int) -> list:
		"""
		Return the list of all the actions that can be executed in state `s`.

		Returns
		-------
		output : list of str
			A list of actions.
		"""
		return self.states[s].actions()

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
		return self.states[s1].tau(action,s2,obs)
			
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

			while action not in self.states[current].transition_matrix:
				action = scheduler.getAction()
			
			res.append(action)
			next_state, observation = self.states[current].next(action)
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

		return [seq,val]

	def logLikelihood(self,sequences: list) -> float:
		"""
		Compute the average loglikelihood of a set of sequences.

		Parameters
		----------
		sequences: list containing one list of str and one list of int
			set of sequences of actions-observations.
		
		Returns
		-------
		output: float
			loglikelihood of ``sequences`` under this model.
		"""
		sequences_sorted = sequences[0][:]
		sequences_sorted.sort()
		loglikelihood = 0.0
		alpha_matrix = self._initAlphaMatrix(len(sequences_sorted[0])//2)
		for seq in range(len(sequences_sorted)):
			sequence_actions = [sequences_sorted[seq][i] for i in range(0,len(sequences_sorted[seq]),2)]
			sequence_obs = [sequences_sorted[seq][i+1] for i in range(0,len(sequences_sorted[seq]),2)]
			sequence = sequences_sorted[seq]
			times = sequences[1][sequences[0].index(sequence)]
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

		return loglikelihood/sum(sequences[1])
	
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
		nb_states = len(self.states)
		diff_size = len(alpha_matrix)-1 - len(sequence_obs)
		if diff_size < 0: # alpha_matrix too small
			n = zeros(-diff_size * nb_states).reshape(-diff_size,nb_states)
			alpha_matrix = append(alpha_matrix,n,axis=0)
		elif diff_size > 0: #alpha_matrix too big
			alpha_matrix = vsplit(alpha_matrix,[len(alpha_matrix)-diff_size,nb_states])[0]
		for k in range(common,len(sequence_obs)):
			for s in range(nb_states):
				p = array([self.tau(ss,sequence_actions[k],s,sequence_obs[k]) for ss in range(len(self.states))])
				alpha_matrix[k+1,s] = dot(alpha_matrix[k],p)
		return alpha_matrix

	#-------------------------------------------

	def saveToPrism(self,output_file):
		f = open(output_file,'w',encoding="utf-8")
		f.write("mdp\n")
		f.write("\tmodule "+self.name+"\n")
		f.write("\ts : [0.."+str(len(self.states)-1)+"] init "+str(self.initial_state)+";\n")
		f.write("\tl : [0.."+str(len(self.observations()))+"] init "+str(len(self.observations()))+";\n")
		f.write("\n")
		for s in range(len(self.states)):
			state = self.states[s]
			for a in state.actions():
				f.write("\t["+a+"] s="+str(s)+" -> ")
				f.write(" + ".join([str(state.transition_matrix[a][0][t])+" : (s'="+str(state.transition_matrix[a][1][t])+") & (l'="+str(self.observations().index(state.transition_matrix[a][2][t]))+")" for t in range(len(state.transition_matrix[a][0]))]))
				f.write(";\n")
		f.write("\n")
		f.write("endmodule\n")

		for l in range(len(self.observations())):
			f.write('label "'+self.observations()[l]+'" = l='+str(l)+';\n')
		f.close()


def KLDivergence(m1,m2,test_set):
	pm1 = m1.probasSequences(test_set)
	tot_m1 = sum(pm1)
	pm2 = m2.probasSequences(test_set)
	res = 0.0
	for seq in range(len(test_set)):
		if pm2[seq] <= 0.0:
			print(test_set[seq])
			return None
		if tot_m1 > 0.0 and pm1[seq] > 0.0:
			res += (pm1[seq]/tot_m1)*log(pm1[seq]/pm2[seq])
	return res

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
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	
	l = f.readline()
	c = 0
	while l:
		d = {}
		while l != "*\n":
			a = l[:-1]
			p = [float(i) for i in  f.readline()[:-2].split(' ')]
			s = [int(i) for i in  f.readline()[:-2].split(' ')]
			t = f.readline()[:-2].split(' ')
			d[a] = [p,s,t]
			l = f.readline()
		states.append(MDP_state(d,c))
		c += 1
		l = f.readline()
	return MDP(states,initial_state,name)

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
	A pseudo-randomly generated MDP.
	"""
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	states = []
	for i in range(nb_states):
		dic = {}
		for act in actions:
			dic[act] = [randomProbabilities(len(obs)),s,obs]
		states.append(MDP_state(dic,i))
	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return MDP(states,init,"MDP_random_"+str(nb_states)+"_states")

def MDPFileToPrism(file_path,output_file):
	m = loadMDP(file_path)
	m.saveToPrism(output_file)

def loadPrismMDP(file_path):
	f = open(file_path)
	f.readline()
	f.readline()
	l = f.readline()
	l = l.split(' ')

	states = []
	init = int(l[-1][:-2])
	for i in range(int(l[2][4:-1])+1):
		states.append({})

	l = f.readline()
	while l[:-1] != "endmodule":
		act = l[1]
		state = int(l[l.find('=')+1:l.find('-')-1])
		l = (' '+f.readline()).split('+')
		states[state][act] = []
		states[state][act].append([ float(i[1:i.find(':')-1]) for i in l ]) #add proba
		states[state][act].append([ int(i[i.find('=')+1:i.find(')')]) for i in l ]) #add state

		l = f.readline()

	map_s_o = {}
	l = f.readline()

	while l:
		l = l[:-2]
		if not "goal" in l:
			obs = l[l.find('"')+1:l.rfind('"')]
			obs = obs[0].upper() + obs[1:]
			l = l.split('|')
			s = [int(i[i.rfind('=')+1:]) for i in l]
			for ss in s:
				map_s_o[ss] = obs
		l = f.readline()

	for state in range(len(states)):
		for a in states[state]:
			states[state][a].append( [ map_s_o[states[state][a][1][i]] for i in range(len(states[state][a][1])) ] )


	states = [MDP_state(j,i) for i,j in enumerate(states)]

	m = MDP(states,init,file_path[:-6])
	return m
