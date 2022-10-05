import jajapy as ja
from numpy import array
import stormpy

def modelMDP_gridworld():
	alphabet = ['S','M','G','C','W',"done"]
	actions = list("nsew")
	nb_states = 9
	s0 = ja.MDP_state({'n': [(0,'W',1.0)],
					's': [(3,'M',0.6),(4,'G',0.4)],
					'e': [(1,'M',0.6),(4,'G',0.4)],
					'w': [(0,'W',1.0)]
					},alphabet,nb_states,actions)
	s1 = ja.MDP_state({'n': [(1,'W',1.0)],
					's': [(4,'G',0.8),(3,'M',0.1),(5,'C',0.1)],
					'e': [(2,'G',0.8),(5,'C',0.2)],
					'w': [(0,'S',0.75),(3,'M',0.25)]
					},alphabet,nb_states,actions)
	s2 = ja.MDP_state({'n': [(2,'W',1.0)],
					's': [(4,'G',0.8),(3,'M',0.1),(5,'C',0.1)],
					'e': [(2,'W',1.0)],
					'w': [(1,'M',0.6),(4,'G',0.4)]
					},alphabet,nb_states,actions)
	s3 = ja.MDP_state({'n': [(0,'S',0.75),(1,'M',0.25)],
					's': [(6,'G',0.8),(7,'S',0.2)],
					'e': [(4,'G',0.8),(1,'M',0.1),(7,'S',0.1)],
					'w': [(3,'M',1.0)]
					},alphabet,nb_states,actions)
	s4 = ja.MDP_state({'n': [(1,'M',0.6),(0,'S',0.2),(2,'G',0.2)],
					's': [(7,'S',0.75),(6,'G',0.125),(8,'done',0.125)],
					'e': [(5,'C',1.0)],
					'w': [(3,'M',0.6),(0,'S',0.2),(6,'G',0.2)]
					},alphabet,nb_states,actions)
	s5 = ja.MDP_state({'n': [(2,'G',0.8),(1,'M',0.2)],
					's': [(8,'done',0.6),(7,'S',0.4)],
					'e': [(5,'W',1.0)],
					'w': [(4,'G',0.8),(1,'M',0.1),(7,'S',0.1)]
					},alphabet,nb_states,actions)
	s6 = ja.MDP_state({'n': [(3,'M',0.6),(4,'G',0.4)],
					's': [(6,'W',1.0)],
					'e': [(7,'S',0.75),(4,'G',0.25)],
					'w': [(6,'W',1.0)]
					},alphabet,nb_states,actions)
	s7 = ja.MDP_state({'n': [(1,'M',0.6),(0,'S',0.2),(2,'G',0.2)],
					's': [(7,'W',1.0)],
					'e': [(8,'done',0.6),(5,'C',0.4)],
					'w': [(6,'G',0.8),(3,'M',0.2)]
					},alphabet,nb_states,actions)
	s8 = ja.MDP_state({'n': [(8,'done',1.0)],
					's': [(8,'done',1.0)],
					'e': [(8,'done',1.0)],
					'w': [(8,'done',1.0)]
					},alphabet,nb_states,actions)
	matrix = array([s0,s1,s2,s3,s4,s5,s6,s7,s8])
	return ja.MDP(matrix,alphabet,actions,initial_state=0,name="grid world")

def example_3():
	original_model = modelMDP_gridworld()
	# SETS GENERATION
	#----------------
	# We generate 1000 sequences of 10 observations for each set
	scheduler = ja.UniformScheduler(original_model.getActions())
	training_set = original_model.generateSet(1000,10,scheduler)
	test_set = original_model.generateSet(1000,10,scheduler)

	# LEARNING
	#---------
	learning_rate = 0
	output_model = ja.Active_BW_MDP().fit(training_set,learning_rate,
										  nb_iterations=20, nb_sequences=50,
										  epsilon_greedy=0.75, nb_states=9,
										  stormpy_output=False)
	output_quality = output_model.logLikelihood(test_set)
	
	print(output_model)
	print(output_quality)

	# MODEL CHECKING
	#---------------
	storm_model = ja.modeltoStorm(output_model)
	print(storm_model)
	formula_str = "Rmax=? [ F \"done\" ]"
	properties = stormpy.parse_properties(formula_str)
	result = stormpy.check_model_sparse(storm_model, properties[0], extract_scheduler=True)
	scheduler = result.scheduler
	print(result)
	for state in storm_model.states:
		choice = scheduler.get_choice(state)
		action = choice.get_deterministic_choice()
		print("In state {} choose action {}".format(state, output_model.actions[action]))

if __name__ == '__main__':
	example_3()