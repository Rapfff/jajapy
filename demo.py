from jajapy import *
def modelMDP_bigstreet(p=0.75):
	labelling = ['R','L','R','L','OK','HIT']
	transitions = [(0,'m',1,p),(0,'m',2,1-p),(0,'s',3,p),(0,'s',0,1-p),
				   (1,'m',0,p),(1,'m',3,1-p),(1,'s',2,p),(1,'s',1,1-p),
				   (2,'m',5,1.0),(2,'s',4,1.0),(3,'m',5,1.0),(3,'s',4,1.0),
				   (4,'m',4,1.0),(4,'s',4,1.0),(5,'m',5,1.0),(5,'s',5,1.0)]
	return createMDP(transitions,labelling,0,"bigstreet")

m = modelMDP_bigstreet()

initial_model   = loadMDP("jajapy/tests/materials/mdp/random_MDP.txt")
training_set    = loadSet("jajapy/tests/materials/mdp/training_set_MDP.txt")
output_gotten   = Active_BW_MDP().fit(training_set, sul=m, initial_model=initial_model, lr=0,
									  nb_iterations=10,nb_sequences=10,sequence_length=10,
									  stormpy_output=False,output_file="jajapy/tests/materials/mdp/active_output_MDP.txt")