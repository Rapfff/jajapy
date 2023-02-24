import unittest
from ..gohmm import *
from os import remove
from ..base.Set import *
from ..base.BW import BW
from math import exp, sqrt, pi

def modelGoHMM():
	initial_state = [0.1,0.7,0.0,0.0,0.2]
	transitions = [(0,0,0.9),(0,1,0.1),(1,0,0.05),(1,1,0.9),(1,2,0.04),
				   (1,4,0.01),(2,1,0.05),(2,2,0.8),(2,3,0.14),(2,4,0.01),
				   (3,2,0.05),(3,3,0.95),(4,1,0.1),(4,4,0.9)]
	output = [(0,3.0,5.0),(0,2.0,4.0),(1,0.5,1.5),(1,2.5,1.5),(2,0.2,0.7),
			  (2,1.0,1.0),(3,0.0,0.3),(3,1.5,5.0),(4,2.0,4.0),(4,0.5,0.5)]
	return createGoHMM(transitions,output,initial_state,name="My_GoHMM")

m = modelGoHMM()

class GoHMMTestclass(unittest.TestCase):

	def test_GoHMM_state(var):
		mu = m.mu(0)
		b = exp(-0.02)/(5.0*sqrt(2*pi))*exp(-1/32)/(4.0*sqrt(2*pi))
		sigma = [m.output[0][i][1] for i in range(2)]
		
		var.assertEqual(m.a(0,0),0.9)
		var.assertEqual(m.a(0,2),0.0)
		var.assertEqual(sigma,[5.0,4.0])
		var.assertTrue(all(mu==array([3.0,2.0])))
		var.assertEqual(m.b(0,[4.0,3.0]),b)
		var.assertEqual(m.tau(0,0,[4.0,3.0]),0.9*b)
	
	def test_GoHMM_save_load_str(var):
		m.save("test_save.txt")
		mprime = loadGoHMM("test_save.txt")
		var.assertEqual(type(m),type(mprime))
		var.assertEqual(type(m),GoHMM)
		
		var.assertEqual(str(m),str(mprime))
		remove("test_save.txt")
	
	def test_GoHMM_Set(var):
		set1 = m.generateSet(50,10)
		var.assertEqual(set1.type,3)
		set2 = m.generateSet(50,1/4,"geo",6)
		set1.addSet(set2)
		set1.save("test_save.txt")
		set2 = loadSet("test_save.txt")
		var.assertEqual(set2.type,3)
		var.assertTrue(set1.isEqual(set2))
		remove("test_save.txt")
	
	def test_BW_GoHMM(var):
		initial_model   = loadGoHMM("jajapy/tests/materials/gohmm/random_GoHMM.txt")
		training_set    = loadSet("jajapy/tests/materials/gohmm/training_set_GoHMM.txt")
		output_expected = loadGoHMM("jajapy/tests/materials/gohmm/output_GoHMM.txt")
		output_gotten   = BW().fit( training_set, initial_model)
		test_set = m.generateSet(10000,10)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))

if __name__ == "__main__":
	unittest.main()