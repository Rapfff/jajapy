import jajapy as ja
from numpy import array
import stormpy

def modelMC_KnuthDie(p=0.5):
	alphabet = ["P","F","one","two","three","four","five","six"]
	nb_states = 13
	s0 = ja.MC_state([(1 ,'P',p),(2 ,'F',1-p)],alphabet,nb_states)
	s1 = ja.MC_state([(3 ,'P',p),(4 ,'F',1-p)],alphabet,nb_states)
	s2 = ja.MC_state([(5 ,'P',p),(6 ,'F',1-p)],alphabet,nb_states)
	s3 = ja.MC_state([(1 ,'P',p),(7 ,'F',1-p)],alphabet,nb_states)
	s4 = ja.MC_state([(8 ,'P',p),(9 ,'F',1-p)],alphabet,nb_states)
	s5 = ja.MC_state([(10,'P',p),(11,'F',1-p)],alphabet,nb_states)
	s6 = ja.MC_state([(12,'P',p),(2 ,'F',1-p)],alphabet,nb_states)
	s7 = ja.MC_state([(7 ,  'one',1.0)],alphabet,nb_states)
	s8 = ja.MC_state([(8 ,  'two',1.0)],alphabet,nb_states)
	s9 = ja.MC_state([(9 ,'three',1.0)],alphabet,nb_states)
	s10= ja.MC_state([(10, 'four',1.0)],alphabet,nb_states)
	s11= ja.MC_state([(11, 'five',1.0)],alphabet,nb_states)
	s12= ja.MC_state([(12,  'six',1.0)],alphabet,nb_states)
	matrix = array([s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12])
	return ja.MC(matrix,alphabet,initial_state=0,name="Knuth's Die")

def firstGuess():
	alphabet = ["P","F","one","two","three","four","five","six"]
	nb_states = 13
	s0 = ja.MC_state(list(zip([1,2],['P','F'],ja.randomProbabilities(2))),alphabet,nb_states)
	s1 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6],
							  ['P','F','P','F','P','F','P','F','P','F','P','F'],
							  ja.randomProbabilities(12))),
					 alphabet,nb_states)
	s2 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6],
							  ['P','F','P','F','P','F','P','F','P','F','P','F'],
							  ja.randomProbabilities(12))),
					 alphabet,nb_states)
	s3 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12],
							  ['P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',
							  'P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',],
							  ja.randomProbabilities(24))),
					 alphabet,nb_states)

	s4 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12],
							  ['P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',
							  'P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',],
							  ja.randomProbabilities(24))),
					 alphabet,nb_states)

	s5 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12],
							  ['P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',
							  'P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',],
							  ja.randomProbabilities(24))),
					 alphabet,nb_states)

	s6 = ja.MC_state(list(zip([1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12],
							  ['P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',
							  'P','F','P','F','P','F','P','F','P','F','P','F','P','F','P','F',],
							  ja.randomProbabilities(24))),
					 alphabet,nb_states)
	s7 = ja.MC_state([(7 ,  'one',1.0)],alphabet,nb_states)
	s8 = ja.MC_state([(8 ,  'two',1.0)],alphabet,nb_states)
	s9 = ja.MC_state([(9 ,'three',1.0)],alphabet,nb_states)
	s10= ja.MC_state([(10, 'four',1.0)],alphabet,nb_states)
	s11= ja.MC_state([(11, 'five',1.0)],alphabet,nb_states)
	s12= ja.MC_state([(12,  'six',1.0)],alphabet,nb_states)
	matrix = array([s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12])
	return ja.MC(matrix,alphabet,initial_state=0,name="first guess")


def example_4():
	original_model = modelMC_KnuthDie()
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
		current_model = ja.BW_MC().fit(training_set,initial_model=firstGuess())
		current_quality = current_model.logLikelihood(test_set)
		if quality_best < current_quality: #we keep the best model only
				quality_best = current_quality
				best_model = current_model
	
	print(quality_best)
	print(best_model)

	# MODEL CHECKING
	#---------------
	model_storm = ja.modeltoStorm(best_model)
	formula_str = 'P=? [F "five"]'
	properties = stormpy.parse_properties(formula_str)
	result = stormpy.check_model_sparse(model_storm,properties[0])
	print(result.at(model_storm.initial_states[0]))


if __name__ == '__main__':
	example_4()