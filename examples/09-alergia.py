import jajapy as ja
import stormpy

def example_9():
	# MODEL CREATION
	#----------------
	# State 0 is labelled with B, state 1 with T, etc...
	labelling = list("BTSXSPTXPVVE")
	initial_state = 0
	name = "MC_REBER"
	# From state 0 we move to state 1 with probability 0.5
	# and to state 5 with probability 0.5, and so on...
	transitions = [(0,1,0.5),(0,5,0.5),(1,2,0.6),(1,3,0.4),(2,2,0.6),(2,3,0.4),
				   (3,7,0.5),(3,4,0.5),(4,11,1.0),(5,6,0.7),(5,9,0.3),
				   (6,6,0.7),(6,9,0.3),(7,6,0.7),(7,9,0.3),(8,7,0.5),(8,4,0.5),
				   (9,8,0.5),(9,10,0.5),(10,11,1.0),(11,11,1.0)]
	original_model = ja.createMC(transitions,labelling,initial_state,name)

	# TRAINING SET GENERATION
	#------------------------
	# We generate 1000 sequences of 10 observations
	training_set = original_model.generateSet(10000,10)
	test_set     = original_model.generateSet(10000,10)

	# Alergia learning
	#-----------------
	alergia_model = ja.Alergia().fit(training_set,alpha=0.1,stormpy_output=False)

	print(original_model.nb_states)
	print(alergia_model.nb_states)
	

	# Loglikelihood comparison
	#-------------------------
	print('Loglikelihood for the original model  :',original_model.logLikelihood(test_set))
	print('Loglikelihood for Alergia output model:',alergia_model.logLikelihood(test_set))

	# Properties comparison
	#---------------------

	alergia_model = alergia_model.toStormpy()
	original_model = original_model.toStormpy()

	formulas = ['P=? [ G !"P"]','P=? [ G !"X"]','P=? [ F<=5 "P"]']
	for formula in formulas:
		properties = stormpy.parse_properties(formula)
		result_original = stormpy.check_model_sparse(original_model,properties[0])
		result_alergia   = stormpy.check_model_sparse(alergia_model,properties[0])
		print(formula+':')
		print("In the original model:",result_original.at(original_model.initial_states[0]))
		print("In the Alergia output model:",result_alergia.at(alergia_model.initial_states[0]))

if __name__ == "__main__":
	example_9()