import jajapy as ja
import stormpy

def example_10():
	# MODEL CREATION
	#----------------
	actions = list('nsew')
	labels = list("SMGCW")+["GOAL"]
	original_model = ja.loadPrism('materials/grid_3x3.sm')
	original_model.actions = actions

	# TRAINING SET GENERATION
	#------------------------
	# We generate 1000 sequences of 10 observations
	# using an uniform scheduler to resolve the non-deterministic choices.
	training_set = original_model.generateSet(1000,10,scheduler=ja.UniformScheduler(actions))
	test_set     = original_model.generateSet(1000,10,scheduler=ja.UniformScheduler(actions))


	# IOAlergia learning
	#-----------------
	ioalergia_model = ja.IOAlergia().fit(training_set,epsilon=0.5,stormpy_output=False)

	print(original_model.nb_states)
	print(ioalergia_model.nb_states)
	

	# Loglikelihood comparison
	#-------------------------
	print('Loglikelihood for the original model  :',original_model.logLikelihood(test_set))
	print('Loglikelihood for IOAlergia output model:',ioalergia_model.logLikelihood(test_set))
	
	# Properties comparison
	#-------------------------
	original_model = ja.jajapyModeltoStormpy(original_model)
	ioalergia_model = ja.jajapyModeltoStormpy(ioalergia_model)

	formulas = ["Pmax=? [ F<=3 \"GOAL\"  ]","Pmax=? [ !(\"C\"|\"W\") U<=6\"GOAL\" ]", "Pmax=? [ F<=5 \"GOAL\"  ]"]
	
	for formula in formulas:
		properties = stormpy.parse_properties(formula)
		result_original = stormpy.check_model_sparse(original_model, properties[0])
		result_original = result_original.at(original_model.initial_states[0])
		result_ioalergia = stormpy.check_model_sparse(ioalergia_model, properties[0])
		result_ioalergia = result_ioalergia.at(ioalergia_model.initial_states[0])
		print(formula,'in the original model:',str(result_original))
		print(formula,'in the output model active:',str(result_ioalergia))
		print()


if __name__ == "__main__":
	example_10()