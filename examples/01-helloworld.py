import jajapy as ja
import stormpy

def example_1():
	# MODEL CREATION
	#----------------
	# State 0 is labelled with H, state 1 with E, etc...
	labeling = list('HELOWRD')
	initial_state = 0 # we always start in state 0.
	name = "MC_Helloworld"
	# From state 0 we move to state 1 with probability 1.0,
	# from state 2 we move to state 3 with probability 1/3, etc...
	transitions = [(0,1,1.0),(1,2,1.0),(2,2,1/3),(2,3,1/3),(2,6,1/3),
				   (3,4,0.5),(3,5,0.5),(4,3,1.0),(5,6,1.0),(6,6,1.0)]

	original_model = ja.createMC(transitions,labeling,initial_state,name)
	print(original_model)
	# SETS GENERATION
	#----------------
	# We generate 1000 sequences of 10 observations for each set.
	training_set = original_model.generateSet(1000,10)

	# LEARNING
	#---------
	output_model = ja.BW_MC().fit(training_set,nb_states=7)
	
	# EVALUATION
	#-----------
	# We convert the original model to a Stormpy one,
	# to compare the model checking results.
	original_model = ja.jajapyModeltoStormpy(original_model)
	# Now we can model check the two models, using Stormpy.
	
	formula_str = 'P=? [ (((("init" U "H") U "E") U "L") U=5 "O")]'
	properties = stormpy.parse_properties(formula_str)
	result_original = stormpy.check_model_sparse(original_model,properties[0])
	result_output   = stormpy.check_model_sparse(output_model,properties[0])
	
	print('Probability that the original model generates HELLO:')
	print(result_original.at(original_model.initial_states[0]))
	print('Probability that the output model generates HELLO:')
	print(result_output.at(output_model.initial_states[0]))

	formula_str = 'P=? [ F<=10 "D" ]'
	
	properties = stormpy.parse_properties(formula_str)
	result_original = stormpy.check_model_sparse(original_model,properties[0])
	result_output   = stormpy.check_model_sparse(output_model,properties[0])
	
	print("Model checking result for "+formula_str+' in the original model:')
	print(result_original.at(original_model.initial_states[0]))
	print("Model checking result for "+formula_str+' in the output model:')
	print(result_output.at(output_model.initial_states[0]))



if __name__ == '__main__':
	example_1()