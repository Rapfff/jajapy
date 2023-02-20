import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import jajapy as ja
import stormpy

def example_5():
	# MODEL CREATION
	#----------------
	# The first state is labeled with red, etc...
	labelling = ['red','red','yellow','blue','blue']
	# We move from state 0 to state 1 with a rate of 0.08, and so on...
	transitions = [(0,1,0.08),(0,2,0.12),(1,2,1.0),
				(2,0,0.2),(2,3,0.1),(2,4,0.2),
				(3,1,0.5),(3,4,0.5),(4,2,0.25)]
	original_model = ja.createCTMC(transitions,labelling,initial_state=0,name="My_CTMC")

	# SETS GENERATION
	#----------------
	# We generate 1000 sequences of 10 observations for each set,
	# including the dwell times.
	training_set = original_model.generateSet(1000,10,timed=True)
	test_set = original_model.generateSet(1000,10,timed=True)

	# ESTIMATING THE labelling
	#------------------------

	nb_trials = 10
	best_model = None
	quality_best = -1024
	for n in range(1,nb_trials+1):
		current_model = ja.CTMC_random(nb_states=5,
						labelling=['red','yellow','blue'],
						self_loop=False,
						random_initial_state=True,
						min_exit_rate_time=1.0,
						max_exit_rate_time=5.0)
		current_quality = current_model.logLikelihood(test_set)
		if quality_best < current_quality: #we keep the best model only
				quality_best = current_quality
				best_model = current_model

	print(best_model.labelling)
	
	output_model = ja.BW().fit(training_set,initial_model=best_model)
	
	# EVALUATION
	#-----------
	# We convert the original model to a Stormpy one,
	# to compare the model checking results.
	original_model = ja.jajapyModeltoStormpy(original_model)
	formulas = ["T=? [ F \"blue\"  ]", "P=? [ F>5 \"blue\"  ]"]
	for formula in formulas:
		properties = stormpy.parse_properties(formula)
		result_original = stormpy.check_model_sparse(original_model, properties[0])
		result_original = result_original.at(original_model.initial_states[0])
		result_output = stormpy.check_model_sparse(output_model, properties[0])
		result_output = result_output.at(output_model.initial_states[0])
		print(formula,'in the original model:',str(result_original))
		print(formula,'in the output model active:',str(result_output))
		print()
	
if __name__ == '__main__':
	example_5()