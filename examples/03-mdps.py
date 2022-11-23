import jajapy as ja
import stormpy


def modelMDP_gridworld():
	labeling = ['S','M','G',
				'M','G','C',
				'G','S',"done",
				'W','W','W','W','W','W','W']
	actions = list("nsew")
	nb_states = 9
	transitions=[(0,'n',9,1.0),(0,'s',3,0.6),(0,'s',5,0.4),(0,'e',1,0.6),(0,'e',5,0.4),(0,'w',9,1.0),
		(9,'n',9,1.0),(9,'s',3,0.6),(9,'s',5,0.4),(9,'e',1,0.6),(9,'e',5,0.4),(9,'w',9,1.0),
		(1,'n',10,1.0),(1,'s',4,0.8),(1,'s',3,0.1),(1,'s',5,0.1),(1,'e',2,0.8),(1,'e',5,0.2),(1,'w',0,0.75),(1,'w',3,0.25),
		(10,'n',10,1.0),(10,'s',4,0.8),(10,'s',3,0.1),(10,'s',5,0.1),(10,'e',2,0.8),(10,'e',5,0.2),(10,'w',0,0.75),(10,'w',3,0.25),
		(2,'n',11,1.0),(2,'s',5,1.0),(2,'e',11,1.0),(2,'w',1,0.6),(2,'w',4,0.4),
		(11,'n',11,1.0),(11,'s',5,1.0),(11,'e',11,1.0),(11,'w',1,0.6),(11,'w',4,0.4),
		(3,'n',0,0.75),(3,'n',1,0.25),(3,'s',6,0.8),(3,'s',7,0.2),(3,'e',4,0.8),(3,'e',1,0.1),(3,'e',7,0.1),(3,'w',12,1.0),
		(12,'n',0,0.75),(12,'n',1,0.25),(12,'s',6,0.8),(12,'s',7,0.2),(12,'e',4,0.8),(12,'e',1,0.1),(12,'e',7,0.1),(12,'w',12,1.0),
		(4,'n',1,0.6),(4,'n',0,0.2),(4,'n',2,0.2),(4,'s',7,0.75),(4,'s',6,0.125),(4,'s',8,0.125),(4,'e',5,1.0),(4,'w',3,0.6),(4,'w',0,0.2),(4,'w',6,0.2),
		(5,'n',2,0.8),(5,'n',1,0.2),(5,'s',8,0.6),(5,'s',7,0.4),(5,'e',13,1.0),(5,'w',4,0.8),(5,'w',1,0.1),(5,'w',7,0.1),
		(13,'n',2,0.8),(13,'n',1,0.2),(13,'s',8,0.6),(13,'s',7,0.4),(13,'e',13,1.0),(13,'w',4,0.8),(13,'w',1,0.1),(13,'w',7,0.1),
		(6,'n',3,0.6),(6,'n',4,0.4),(6,'s',14,1.0),(6,'e',7,0.75),(6,'e',4,0.25),(6,'w',14,1.0),
		(14,'n',3,0.6),(14,'n',4,0.4),(14,'s',14,1.0),(14,'e',7,0.75),(14,'e',4,0.25),(14,'w',14,1.0),
		(7,'n',4,0.8),(7,'n',3,0.1),(7,'n',5,0.1),(7,'s',15,1.0),(7,'e',8,0.6),(7,'e',4,0.4),(7,'w',6,0.8),(7,'w',3,0.2),
		(15,'n',4,0.8),(15,'n',3,0.1),(15,'n',5,0.1),(15,'s',15,1.0),(15,'e',8,0.6),(15,'e',4,0.4),(15,'w',6,0.8),(15,'w',3,0.2),
		(8,'n',8,1.0),(8,'s',8,1.0),(8,'e',8,1.0),(8,'w',8,1.0)]
	return ja.createMDP(transitions,labeling,initial_state=0,name="grid world")

def example_3():
	original_model = modelMDP_gridworld()
	
	# SETS GENERATION
	#------------------------
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
	storm_model = ja.jajapyModeltoStormpy(output_model)
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