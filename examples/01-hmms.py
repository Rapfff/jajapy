import jajapy as ja
from numpy import array


def example_1():
	alphabet = ['a','b','x','y']
	nb_states = 5

	# MODEL CREATION
	#----------------
	# in the next state we generate 'x' with probability 0.4, and 'y' with probability 0.6
	# once an observation generated, we move to state 1 or 2 with probability 0.5
	# the id of this state is 0.
	s0 = ja.HMM_state([("x",0.4),("y",0.6)],[(1,0.5),(2,0.5)],alphabet,nb_states)
	s1 = ja.HMM_state([("a",0.8),("b",0.2)],[(3,1.0)],alphabet,nb_states)
	s2 = ja.HMM_state([("a",0.1),("b",0.9)],[(4,1.0)],alphabet,nb_states)
	s3 = ja.HMM_state([("x",0.5),("y",0.5)],[(0,0.8),(1,0.1),(2,0.1)],alphabet,nb_states)
	s4 = ja.HMM_state([("y",1.0)],[(3,1.0)],alphabet,nb_states)
	transitions = array([s0[0],s1[0],s2[0],s3[0],s4[0]])
	output = array([s0[1],s1[1],s2[1],s3[1],s4[1]])
	original_model = ja.HMM(transitions,output,alphabet,initial_state=0,name="My HMM")
	print(original_model)
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
	output_model = ja.BW_HMM().fit(training_set, nb_states=5)
	print(output_model)

	# OUTPUT EVALUATION
	#------------------
	# We generate 1000 sequences of 10 observations
	test_set = original_model.generateSet(set_size=1000, param=10)
	ll_original = original_model.logLikelihood(test_set)
	ll_output   =   output_model.logLikelihood(test_set)
	quality = ll_original - ll_output
	print(quality)

if __name__ == "__main__":

	example_1()