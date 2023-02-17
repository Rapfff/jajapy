import jajapy as ja
import stormpy

def example_4():
	# MODEL CREATION
	#----------------
	original_model = ja.loadPrism('materials/grid_3x3.sm')
	actions = list('nsew')
	labels = list("SMGCW")+["GOAL"]
	original_model.actions = actions # otherwise the actions are a0, a1, etc...
	# SETS GENERATION
	#------------------------
	# We generate 1000 sequences of 10 observations for each set, 
	# using an uniform scheduler to resolve the non-deterministic choices
	# then we merge the two sets.
	training_set = original_model.generateSet(1000,10,scheduler=ja.UniformScheduler(actions))

	# INITIAL HYPOTHESIS GENERATION
	#------------------------------
	initial_hypothesis = ja.MDP_random(nb_states=16,labeling=labels,
									actions=actions,random_initial_state=False)
	initial_hypothesis.labeling = original_model.labeling

	# LEARNING
	#---------
	output_model_active  = ja.Active_BW_MDP().fit(training_set, lr=0, nb_iterations=10,
												nb_sequences=100,initial_model=initial_hypothesis)

	formulas = ["Pmax=? [ F<=3 \"GOAL\"  ]","Pmax=? [ !(\"C\"|\"W\") U<=6\"GOAL\" ]", "Pmax=? [ F<=5 \"GOAL\"  ]"]
	original_model = ja.jajapyModeltoStormpy(original_model)
	for formula in formulas:
		properties = stormpy.parse_properties(formula)
		result_original = stormpy.check_model_sparse(original_model, properties[0])
		result_original = result_original.at(original_model.initial_states[0])
		result_output_active = stormpy.check_model_sparse(output_model_active, properties[0])
		result_output_active = result_output_active.at(output_model_active.initial_states[0])
		print(formula,'in the original model:',str(result_original))
		print(formula,'in the output model active:',str(result_output_active))
		print()


if __name__ == '__main__':
	example_4()