Examples
========
In this example, we will:
1. Create a HMM *H* from scratch,
2. Use it to generate a training set,
3. Use the Baum-Welch algorithm to learn, from the training set, *H*,
4. Compare *H* with the model generated at the previous step.

Creating a HMM
--------------
We can create the model depicted here (**TODO**) like this:
``
import jajapy as ja
# in the next state we generate 'x' with probaility 0.4, and 'y' with probability 0.6
# once an observation generated, we move to state 1 or 2 with probability 0.5
# the id of this state is 0.
s0 = ja.HMM_state([[0.4,0.6],['x','y']],
				  [[0.5,0.5],[1,2]],
				  0)
s1 = ja.HMM_state([[0.8,0.2],['a','b']],
				  [[1.0],[3]],
				  1)
s2 = ja.HMM_state([[0.1,0.9],['a','b']],
				  [[1.0],[4]],
				  2)
s3 = ja.HMM_state([[0.5,0.5],['x','y']],
				  [[0.8,0.1,0.1],[0,1,2]],
				  3)
s4 = ja.HMM_state([[1.0],['y']],
				  [[1.0],[3]],
				  4)
original_model = ja.HMM([h_s0,h_s1,h_s2,h_s3,h_s4],0,"My HMM")
print(original_model)
``
*(optional)* This model can be saved into a text file and then loaded as follow:
``
original_model.save("my_model.txt")

original_model.load("my_model.txt")
``
