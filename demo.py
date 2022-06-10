import jajapy as ja


def modelMC_REBER():
	s0 = ja.MC_state([[1.0],[1],['B']],0)
	s1 = ja.MC_state([[0.5,0.5],[2,3],['T','P']],1)
	s2 = ja.MC_state([[0.6,0.4],[2,4],['S','X']],2)
	s3 = ja.MC_state([[0.7,0.3],[3,5],['T','V']],3)
	s4 = ja.MC_state([[0.5,0.5],[3,6],['X','S']],4)
	s5 = ja.MC_state([[0.5,0.5],[4,6],['P','V']],5)
	s6 = ja.MC_state([[1.0],[6],['E']],6)
	return ja.MC([s0,s1,s2,s3,s4,s5,s6],0,"MC_REBER")

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
