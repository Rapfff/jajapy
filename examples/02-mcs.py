import jajapy as ja

def example_2():
	# MODEL CREATION
	#----------------
	# State 0 is labelled with B, state 1 with T, etc...
	labeling = list("BTSXSPTXPVVE")
	initial_state = 0
	name = "MC_REBER"
	# From state 0 we move to state 1 with probability 0.5
	# and to state 5 with probability 0.5
	transitions = [(0,1,0.5),(0,5,0.5),(1,2,0.6),(1,3,0.4),(2,2,0.6),(2,3,0.4),
				   (3,7,0.5),(3,4,0.5),(4,11,1.0),(5,6,0.7),(5,9,0.3),
				   (6,6,0.7),(6,9,0.3),(7,6,0.7),(7,9,0.3),(8,7,0.5),(8,4,0.5),
				   (9,8,0.5),(9,10,0.5),(10,11,1.0),(11,11,1.0)]

	original_model = ja.createMC(transitions,labeling,initial_state,name)
	
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
		current_model = ja.BW_MC().fit(training_set,nb_states=12,pp=n, stormpy_output=False)
		current_quality = current_model.logLikelihood(test_set)
		if quality_best < current_quality: #we keep the best model only
				quality_best = current_quality
				best_model = current_model
	
	print(quality_best)
	print(best_model)


if __name__ == '__main__':
	example_2()