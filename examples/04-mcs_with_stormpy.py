import jajapy as ja
import stormpy

def modelMC_KnuthDie(p=0.5):
	labeling = ['','H','T','H','T','H','T','T','H','T','H','T','H',"one","two","three","four","five","six"]
	initial_state = 0
	name="Knuth's Die"
	transitions = [(0,1,p),(0,2,1-p),(1,3,p),(1,4,1-p),(2,5,p),(2,6,1-p),
				   (3,1,p),(3,7,1-p),(4,8,p),(4,9,1-p),(5,10,p),(5,11,1-p),
				   (6,12,p),(6,2,1-p),(7,13,1.0),(8,14,1.0),(9,15,1.0),
				   (10,16,1.0),(11,17,1.0),(12,18,1.0)]
	return ja.createMC(transitions,labeling,initial_state,name)

def firstGuess():
	labeling = ['','H','T','H','T','H','T','T','H','T','H','T','H',"one","two","three","four","five","six"]
	initial_state = 0
	name="first guess"
	p = ja.randomProbabilities(2)
	transitions = [(0,1,p[0]),(0,2,p[2])]

	for src in range(1,7):
		p = ja.randomProbabilities(6)
		for dest in range(1,7):
			transitions.append((src,dest,p[dest-1]))

	transitions += [(7,13,1.0),(8,14,1.0),(9,15,1.0),
				   (10,16,1.0),(11,17,1.0),(12,18,1.0)]
	return ja.createMC(transitions,labeling,initial_state,name)


def example_4():
	original_model = modelMC_KnuthDie()
	# SETS GENERATION
	#----------------
	# We generate 1000 sequences of 10 observations for each set
	training_set = original_model.generateSet(1000,10)
	test_set = original_model.generateSet(1000,10)

	# LEARNING
	#---------
	nb_trials = 10 # we will repeat learn this model 10 times
	best_model = None
	quality_best = -1024
	for n in range(1,nb_trials+1):
		current_model = ja.BW_MC().fit(training_set,initial_model=firstGuess(),stormpy_output=False)
		current_quality = current_model.logLikelihood(test_set)
		if quality_best < current_quality: #we keep the best model only
				quality_best = current_quality
				best_model = current_model
	
	print(quality_best)
	print(best_model)

	# MODEL CHECKING
	#---------------
	model_storm = ja.jajapyModeltoStorm(best_model)
	formula_str = 'P=? [F "five"]'
	properties = stormpy.parse_properties(formula_str)
	result = stormpy.check_model_sparse(model_storm,properties[0])
	print(result.at(model_storm.initial_states[0]))


if __name__ == '__main__':
	example_4()