import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import jajapy as ja
import stormpy

def example_3():
	# MODEL CREATION
	#----------------
	original_model = ja.loadPrism('materials/grid_4x4.sm')
	actions = list('nsew')
	labels = list("SMGCW")+["GOAL"]
	original_model.actions = actions # otherwise the actions are a0, a1, etc...
	# SETS GENERATION
	#------------------------
	# We generate 1000 sequences of 10 observations for each set, 
	# using an uniform scheduler to resolve the non-deterministic choices
	training_set = original_model.generateSet(1000,30,scheduler=ja.UniformScheduler(actions))

	# INITIAL HYPOTHESIS GENERATION
	#------------------------------
	initial_hypothesis = ja.MDP_random(nb_states=26,labelling=labels,actions=actions,random_initial_state=False)
	initial_hypothesis.labelling = original_model.labelling

	# LEARNING
	#---------
	output_model = ja.BW().fit(training_set,initial_model=initial_hypothesis)

	formulas = ["Pmax=? [ F<=5 \"GOAL\"  ]","Pmax=? [ !(\"C\"|\"W\") U<=8\"GOAL\" ]", "Pmax=? [ F<=12 \"GOAL\"  ]"]
	original_model = ja.jajapyModeltoStormpy(original_model)
	for formula in formulas:
		properties = stormpy.parse_properties(formula)
		result_original = stormpy.check_model_sparse(original_model, properties[0])
		result_original = result_original.at(original_model.initial_states[0])
		result_output = stormpy.check_model_sparse(output_model, properties[0])
		result_output = result_output.at(output_model.initial_states[0])
		print(formula,'in the original model:',str(result_original))
		print(formula,'in the output   model:',str(result_output))
		print()


if __name__ == '__main__':
	example_3()