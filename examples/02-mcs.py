import jajapy as ja
from numpy import array

def example_2():
	alphabet = list("BTPSXVE")
	initial_state = 0
	nb_states = 7

	# MODEL CREATION
	#----------------
	# in the next state we generate 'B' while moving to state 1
	# with probability 1.0.
	s0 = ja.MC_state([(1,'B',1.0)],alphabet,nb_states)
	s1 = ja.MC_state([(2,'T',0.5),(3,'P',0.5)],alphabet,nb_states)
	s2 = ja.MC_state([(2,'S',0.6),(4,'X',0.4)],alphabet,nb_states)
	s3 = ja.MC_state([(3,'T',0.7),(5,'V',0.3)],alphabet,nb_states)
	s4 = ja.MC_state([(3,'X',0.5),(6,'S',0.5)],alphabet,nb_states)
	s5 = ja.MC_state([(4,'P',0.5),(6,'V',0.5)],alphabet,nb_states)
	s6 = ja.MC_state([(6,'E',1.0)],alphabet,nb_states)
	matrix = array([s0,s1,s2,s3,s4,s5,s6])
	original_model = ja.MC(matrix,alphabet,initial_state,"MC_REBER")

	# SETS GENERATION
	#------------------------
	# We generate 1000 sequences of 10 observations for each set
	training_set = original_model.generateSet(1000,10)
	test_set = original_model.generateSet(1000,10)

	# LEARNING
	#---------
	nb_trials = 10 # we will repeat learn this model 10 times
	best_model = None
	quality_best = -1024
	for n in range(1,nb_trials+1):
		current_model = ja.BW_MC().fit(training_set,nb_states=7,pp=n)
		current_quality = current_model.logLikelihood(test_set)
		if quality_best < current_quality: #we keep the best model only
				quality_best = current_quality
				best_model = current_model
	
	print(quality_best)
	print(best_model)


if __name__ == '__main__':
	example_2()