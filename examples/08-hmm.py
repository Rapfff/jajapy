import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import jajapy as ja
from numpy import array


def example_8():

	# MODEL CREATION
	#----------------
	# in the next state (s0) we generate 'x' with probability 0.4, and 'y' with probability 0.6
	# once an observation is generated, we move to state 1 or 2 with probability 0.5
	transitions = [(0,1,0.5),(0,2,0.5),(1,3,1.0),(2,4,1.0),
				   (3,0,0.8),(3,1,0.1),(3,2,0.1),(4,3,1.0)]
	emission = [(0,"x",0.4),(0,"y",0.6),(1,"a",0.8),(1,"b",0.2),
				(2,"a",0.1),(2,"b",0.9),(3,"x",0.5),(3,"y",0.5),(4,"y",1.0)]
	original_model = ja.createHMM(transitions,emission,initial_state=0,name="My HMM")
	
	#original_model.save("my_model.txt")
	#original_model = ja.loadHMM("my_model.txt")
	# TRAINING SET GENERATION
	#------------------------
	# We generate 1000 sequences of 10 observations
	training_set = original_model.generateSet(set_size=1000, param=10)
	#training_set.save("my_training_set.txt")
	#training_set = ja.loadSet("my_training_set.txt")

	# LEARNING
	#---------
	initial_hypothesis = ja.HMM_random(5,alphabet=list("abxy"),random_initial_state=False)
	output_model = ja.BW().fit(training_set, initial_hypothesis)

	# OUTPUT EVALUATION
	#------------------
	# We generate 1000 sequences of 10 observations
	test_set = original_model.generateSet(set_size=1000, param=10)
	ll_original = original_model.logLikelihood(test_set)
	ll_output   =   output_model.logLikelihood(test_set)
	quality = abs(ll_original - ll_output)
	print("loglikelihood distance:",quality)

if __name__ == "__main__":
	example_8()