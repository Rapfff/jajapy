import jajapy as ja
import matplotlib.pyplot as plt
from os import remove
from numpy import dot, array, zeros
from random import random
from multiprocessing import Pool

NB_EXP = 10
TR_DIM = ( 100,20)
TS_DIM = (1000,20)
IT_DIM = (  20,20)
NB_IT = 20

class ActiveLearningScheduler:
	"""
	Class for a scheduler used during the active learning sampling.
	"""
	def __init__(self,mat,m) -> None:
		"""
		Create an ActiveLearningScheduler.

		Parameters
		----------
		m : MDP
			The current hypothesis.
		"""
		self.m = m
		self.mat = mat
		self.nb_states = self.m.nb_states
		self.reset()
		
	def reset(self) -> None:
		"""
		Resets the scheduler. Should be done before genereting a new trace.
		"""
		self.last_act = None
		self.alpha_matrix = array([self.m.pi(s) for s in range(self.nb_states)])
	
	def getActionState(self,s:int) -> str:
		"""
		Returns an action chosen by the scheduler
		according to the expected current state.

		Parameters
		----------
		s : int
			Expected current state.

		Returns
		-------
		str
			Action chosen by the scheduler.
		"""
		tot = self.mat[s].sum()
		acts = [(tot-i)/(tot*(self.m.nb_actions-1)) for i in self.mat[s]]
		return ja.resolveRandom(acts)
		#return argmin(self.mat[s])


	def getAction(self) -> str:
		"""
		Returns the action chosen by the scheduler

		Returns
		-------
		str
			Action chosen by the scheduler.
		"""
		tot = self.alpha_matrix.sum()
		if tot <= 0.0:
			t = [1/len(self.alpha_matrix)]*len(self.alpha_matrix)
		else:
			t = [i/tot for i in self.alpha_matrix]
		s_i = ja.resolveRandom(t)
		self.last_act = self.getActionState(s_i)
		return self.m.actions[self.last_act]


	def addObservation(self,obs: str) -> None:
		"""
		Updates the scheduler according to the given observation ``obs``.

		Parameters
		----------
		obs : str
			An observation.
		"""
		tot = sum(self.alpha_matrix)
		if tot > 0.0:
			for s in range(self.m.nb_states):
				self.mat[s,self.last_act] += self.alpha_matrix[s]/tot
			alpha_matrix = zeros(self.nb_states)
			for s in range(self.nb_states):
				p = array([self.m.tau(ss,self.m.actions[self.last_act],s,obs) for ss in range(self.nb_states)])
				alpha_matrix[s] = dot(self.alpha_matrix,p)
			self.alpha_matrix = alpha_matrix
	
	
class Active_BW_Experiment(ja.BW):
	def __init__(self):
		super().__init__()
	
	def fit(self, sul:ja.MDP, nb_iterations: int,
	 		nb_sequences: int, sequence_length: int,
			test_set: ja.Set, ll_sul:float, initial_model=None, pp: str=''):
		pp = str(pp)
		
		total_traces = ja.loadSet("traces_saved.txt")
		self.sul = sul
		super().fit(total_traces, initial_model,
					pp=pp+" Initial iteration:",stormpy_output=False,
					verbose=2)
		self.h.save("h_saved.txt")

		ll = abs(ll_sul - self.h.logLikelihood(test_set))
		with open("output_passive"+pp+".txt",'w') as f:
			f.write(str(ll)+"\n")
		with open("output_active"+pp+".txt",'w') as f:
			f.write(str(ll)+"\n")
			
		# --- ACTIVE  ---
		c = 1
		while c <= nb_iterations :
			traces = self._addTraces(sequence_length,nb_sequences,total_traces,1.0)
			total_traces.addSet(traces)
			
			super().fit(total_traces, initial_model=self.h,
						pp=pp+" Active iteration "+str(c)+"/"+str(nb_iterations)+": ",stormpy_output=False,verbose=2)
			
			ll = abs(ll_sul - self.h.logLikelihood(test_set))
			with open("output_active"+pp+".txt",'a') as f:
				f.write(str(ll)+"\n")
			c += 1
		
		# --- PASSIVE ---
		self.h = ja.loadMDP("h_saved.txt")
		total_traces = ja.loadSet("traces_saved.txt")
		c = 1
		while c <= nb_iterations :
			traces = self._addTraces(sequence_length,nb_sequences,total_traces,0.0)
			total_traces.addSet(traces)
			
			super().fit(total_traces, initial_model=self.h,
						pp=pp+" Passive iteration "+str(c)+"/"+str(nb_iterations)+": ",stormpy_output=False,verbose=2)
			
			ll = abs(ll_sul - self.h.logLikelihood(test_set))
			with open("output_passive"+pp+".txt",'a') as f:
				f.write(str(ll)+"\n")
			c += 1

		return self.h

	def _addTraces(self,sequence_length: int,nb_sequence: int,
				  traces: ja.Set,epsilon_greedy: float) -> list:
		"""
		Generates an active learning sample.

		Parameters
		----------
		sequence_length : int
			Length of each sequence in the sample.
		nb_sequence : int
			Number of sequences in the sample.
		traces : Set
			Training set until now. Used to compute the active learning
			sampling strategy.
		epsilon_greedy : It will use the active learning scheduler with probability
			``epsilon_greedy`` and use an uniform scheduler with probability
			`1-epsilon_greedy`.
			
		Returns
		-------
		list
			The active learning sample
		"""
		mat = self._strategy(traces)
		scheduler_exploit = ActiveLearningScheduler(mat,self.h)
		scheduler_explore = ja.UniformScheduler(self.h.getActions())

		traces = ja.Set([],t=1)
		for n in range(nb_sequence):			
			if random() > epsilon_greedy:
				seq = self.sul.run(sequence_length,scheduler_explore)
			else:
				seq = self.sul.run(sequence_length,scheduler_exploit)

			if not seq in traces.sequences:
				traces.sequences.append(seq)
				traces.times.append(0)
			traces.times[traces.sequences.index(seq)] += 1
		return traces

	def _strategy(self,traces:ja.Set) -> array:
		if self.processes > 1:
		#if False:
			p = Pool(processes = self.processes)
			tasks = []
			for seq,times in zip(traces.sequences,traces.times):
				tasks.append(p.apply_async(self._computeProbas, [seq, times,]))
			temp = [res.get() for res in tasks if res != False]
		else:
			temp = [self._computeProbas(seq, times) for seq,times in zip(traces.sequences,traces.times)]
			temp = [i for i in temp if i!=False]
		res = sum(temp)
		return res

	def _computeProbas(self,seq:list,times:int) -> array:
		sequence_actions = [seq[i+1] for i in range(0,len(seq)-1,2)]
		sequence_obs = [seq[i] for i in range(0,len(seq)-1,2)] + [seq[-1]]
		alpha_matrix = self._computeAlphas(sequence_obs,sequence_actions)
		beta_matrix  = self._computeBetas( sequence_obs,sequence_actions)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq <= 0.0:
			return False
		res = zeros(shape=(self.nb_states,len(self.actions)))

		for s in range(self.nb_states):
			for i,a in enumerate(self.actions):
				arr_dirak = array([1.0 if t == a else 0.0 for t in sequence_actions])
				res[s,i] += dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1]*arr_dirak,times/proba_seq).sum()
		return res

def experiment(sul,training_set_dim,nb_it,active_iteration_dim,test_set_dim):
	training_set = sul.generateSet(training_set_dim[0],training_set_dim[1],ja.UniformScheduler(sul.actions))
	training_set.save("traces_saved.txt")
	test_set = sul.generateSet(test_set_dim[0],test_set_dim[1],ja.UniformScheduler(sul.actions))
	ll_sul = sul.logLikelihood(test_set)
	for n in range(1,NB_EXP+1):
		initial_hypothesis = ja.MDP_random(sul.nb_states-1,list(range(len(set(sul.labelling))-1)),sul.actions,False,False)
		initial_hypothesis.labelling = sul.labelling
		Active_BW_Experiment().fit(sul,nb_it,active_iteration_dim[0],
								active_iteration_dim[1],test_set,
								ll_sul, initial_hypothesis,pp=n)
	remove("h_saved.txt")
	remove("traces_saved.txt")
	printResults(training_set_dim[0],active_iteration_dim[0],nb_it)

def printResults(training_set_size,it_size,nb_it):
	y1 = zeros((NB_EXP,nb_it+1))
	for n in range(1,NB_EXP+1):
		with open("output_passive"+str(n)+".txt") as f:
			for m in range(nb_it+1):
				y1[n-1,m] = float(f.readline()[:-1])*1000

	y2 = zeros((NB_EXP,nb_it+1))
	for n in range(1,NB_EXP+1):
		with open("output_active"+str(n)+".txt") as f:
			for m in range(nb_it+1):
				y2[n-1,m] = float(f.readline()[:-1])*1000
	
	data_1 = {
		'x': [training_set_size+i*it_size for i in range(0,nb_it+1)],
		'y': y1.mean(axis=0),
		'yerr': y1.std(axis=0),
		'label':"passive learning"}
	data_2 = {
		'x': [training_set_size+i*it_size for i in range(0,nb_it+1)],
		'y': y2.mean(axis=0),
		'yerr': y2.std(axis=0),
		'label':"active  learning"}
	
	_,ax = plt.subplots()
	for data in [data_1, data_2]:
		plt.errorbar(**data, alpha=.75, fmt=':', capsize=3, capthick=1)
		data = {
			'x': data['x'],
			'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
			'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
		plt.fill_between(**data, alpha=.25)
	ax.set(xlabel='number of sequences',ylabel='loglikelihood distance')
	ax.legend()
	plt.savefig("active_vs_passive.png")
		
if __name__ == '__main__':
	sul = ja.loadPrism('materials/grid_4x4.sm')
	sul.actions = list('nsew')
	experiment(sul,TR_DIM,NB_IT,IT_DIM,TS_DIM)
	printResults(TR_DIM[0],IT_DIM[0],NB_IT)