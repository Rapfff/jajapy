import jajapy as ja
from numpy import array

def modelMC_REBER():
	alphabet = list("BTPSXVE")
	initial_state = 0
	nb_states = 7
	s0 = ja.MC_state([(1,'B',1.0)],alphabet,nb_states)
	s1 = ja.MC_state([(2,'T',0.5),(3,'P',0.5)],alphabet,nb_states)
	s2 = ja.MC_state([(2,'S',0.6),(4,'X',0.4)],alphabet,nb_states)
	s3 = ja.MC_state([(3,'T',0.7),(5,'V',0.3)],alphabet,nb_states)
	s4 = ja.MC_state([(3,'X',0.5),(6,'S',0.5)],alphabet,nb_states)
	s5 = ja.MC_state([(4,'P',0.5),(6,'V',0.5)],alphabet,nb_states)
	s6 = ja.MC_state([(6,'E',1.0)],alphabet,nb_states)
	matrix = array([s0,s1,s2,s3,s4,s5,s6])
	return ja.MC(matrix,alphabet,initial_state,"MC_REBER")

original_model = modelMC_REBER()

training_set = original_model.generateSet(100,10)
test_set = original_model.generateSet(100,10)

nb_trials = 10
best_model = None
quality_best = -1024

for n in range(nb_trials):
	current_model = ja.BW_MC().fit(training_set,nb_states=7,pp=n)
	current_quality = current_model.logLikelihood(test_set)
	if quality_best < current_quality:
		quality_best = current_quality
		best_model = current_model

print(best_model)
print(quality_best)
