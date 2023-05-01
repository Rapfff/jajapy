from numpy import loadtxt, float64, ones, array, argmax, max, min
import jajapy as ja
import matplotlib.pyplot as plt

def training_test_set(cols, len_seq = 14):
	"""
	cols are the columns we want to keep
	len_seq indicates how many days are included in one sequence in the training/test set.
	if len_seq is equal to 7, our training set and test set will contains sequences, and each
	sequence will contain on seven days
	"""
	arr = loadtxt('materials/weather_prediction_dataset.csv',delimiter=',',dtype=str)
	nb_distributions = len(cols)
	arr = arr[:,cols]
	print(arr[0])
	training_set = array(arr[1:3289],dtype=float64) # 2000-2009
	test_set = array(arr[3289:],dtype=float64) # 2010

	complete_seq = len(training_set)//len_seq
	#drop = training_set[len(training_set)-len(training_set)%len_seq:]
	training_set = training_set[:len(training_set)-len(training_set)%len_seq]
	training_set = training_set.reshape((complete_seq,len_seq,nb_distributions))
	training_set = ja.Set(training_set,ones(len(training_set)),t=3)

	complete_seq = len(test_set)//len_seq
	#drop = test_set[len(test_set)-len(test_set)%len_seq:]
	test_set = test_set[:len(test_set)-len(test_set)%len_seq]
	test_set = test_set.reshape((complete_seq,len_seq,nb_distributions))
	test_set = ja.Set(test_set,ones(len(test_set)),t=3)

	return training_set, test_set

def testing(m, seq, steps=5):
	"""
	m is our model
	seq is one trace, i.e. one sequences of observations
	steps corresponds to the number of steps we give to our model, the last
	len(seq)-steps will be forecasted by our model.
	"""

	alphas = m._initAlphaMatrix(steps)
	alphas = m._updateAlphaMatrix(seq[:steps],0,alphas)
	alphas = alphas[-1]
	current = argmax(alphas)
	
	forecast = m.run(len(seq)-steps,current)
	
	fig, axs = plt.subplots(2,4)
	
	features = ["cloud_cover","wind_speed","wind_gust","humidity","pressure","temp_mean","temp_min","temp_max"]

	for i in range(8):
		axs[i%2][i//2].plot(range(1,len(seq)+1),[seq[j][i] for j in range(len(seq))], c='b')
		axs[i%2][i//2].plot(range(steps,len(seq)+1),[seq[steps-1][i]]+[forecast[j][i] for j in range(len(forecast))], c='r')
		axs[i%2][i//2].set_title(features[i])
	plt.show()

def example_8():
	training_set, test_set = training_test_set([19,20,21,22,23,27,28,29],7)

	print("Dimension of our training set:",training_set.sequences.shape)
	print()
	print("First sequence in the training set:\n",training_set.sequences[0])

	initial_hypothesis = ja.GoHMM_random(nb_states=15, nb_distributions=8,
				      					 min_mu=-5.0,max_mu=5.0,
										 min_sigma=1.0,max_sigma=5.0)
	
	output_model = ja.BW().fit(training_set,initial_hypothesis)

	for week in range(1,52,25):
		print("\nWeek number",week+1,'\n')
		testing(output_model,test_set.sequences[week],steps=5)



if __name__ == '__main__':
	example_8()