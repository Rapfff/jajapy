import jajapy as ja

def example_5():
	# ORIGINAL MODEL
	#---------------
	original_model = ja.loadPrism("materials/tandem_5.sm")
	original_model.instantiate(["mu1a","mu1b","mu2","kappa"],[0.2,1.8,2.0,4.0])

	# TRAINING SET GENERATION
	#------------------------
	# We generate 1000 sequences of 30 observations, with the dwell times
	training_set = original_model.generateSet(100,30,timed=True)

	# INITIAL HYPOTHESIS
	#-------------------
	initial_hypothesis = ja.loadPrism("materials/tandem_5.sm")

	# LEARNING
	#---------
	output_val = ja.BW_PCTMC().fit_nonInstantiatedParameters(training_set,
														initial_hypothesis,
														stormpy_output=False,
														min_val=0.1,
														max_val=5.0)

	for parameter in output_val.keys():
		print("parameter",parameter,':')
		print("Estimated value:", output_val[parameter],end='')
		print(", real value:",original_model.parameterValue(parameter))
		print()


if __name__ == "__main__":
	example_5()