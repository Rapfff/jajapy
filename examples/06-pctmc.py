import jajapy as ja

def example_6():
	# ORIGINAL MODEL
	#---------------
	original_model = ja.loadPrism("materials/tandem_3.sm")
	original_model.instantiate(["mu1a","mu1b","mu2","kappa"],[0.2,1.8,2.0,4.0])

	# TRAINING SET GENERATION
	#------------------------
	# We generate 1000 sequences of 10 observations, with the dwell times
	training_set = original_model.generateSet(1000,10,timed=True)

	# INITIAL HYPOTHESIS
	#-------------------
	initial_hypothesis = ja.loadPrism("materials/tandem_3.sm")
	
	# LEARNING
	#---------
	output_val = ja.BW().fit_nonInstantiatedParameters(training_set,
														initial_hypothesis,
														min_val=0.1,
														max_val=5.0)

	for parameter in output_val.keys():
		print("parameter",parameter,':')
		print("Estimated value:", round(output_val[parameter],3),end='')
		print(", real value:",original_model.parameterValue(parameter))
		print()


if __name__ == "__main__":
	example_6()