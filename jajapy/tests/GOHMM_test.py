import unittest
from ..gohmm import *
from os import remove
from ..base.Set import *
from math import exp, sqrt, pi

def modelGOHMM():
	nb_states = 5
	s0 = GOHMM_state(list(zip([0,1],[0.9,0.1])),[3.0,5.0],nb_states)
	s1 = GOHMM_state(list(zip([0,1,2,4],[0.05,0.9,0.04,0.01])),[0.5,1.5],nb_states)
	s2 = GOHMM_state(list(zip([1,2,3,4],[0.05,0.8,0.14,0.01])),[0.2,0.7],nb_states)
	s3 = GOHMM_state(list(zip([2,3],[0.05,0.95])),[0.0,0.3],nb_states)
	s4 = GOHMM_state(list(zip([1,4],[0.1,0.9])),[2.0,4.0],nb_states)
	matrix = array([s0[0],s1[0],s2[0],s3[0],s4[0]])
	output = array([s0[1],s1[1],s2[1],s3[1],s4[1]])
	return GOHMM(matrix,output,[0.1,0.7,0.0,0.0,0.2],"GOHMM")

m = modelGOHMM()

class GOHMMTestclass(unittest.TestCase):

	def test_GOHMM_state(var):
		mu = m.mu(0)
		b = exp(-0.02)/(5.0*sqrt(2*pi))
		sigma = m.output[0][1]
		var.assertEqual(m.a(0,0),0.9)
		var.assertEqual(m.a(0,2),0.0)
		var.assertEqual(mu,3.0)
		var.assertEqual(sigma,5.0)
		var.assertEqual(m.b(0,4.0),b)
		var.assertEqual(m.tau(0,0,4.0),0.9*b)
	
	def test_GOHMM_save_load_str(var):
		m.save("test_save.txt")
		mprime = loadGOHMM("test_save.txt")
		var.assertEqual(str(m),str(mprime))
		remove("test_save.txt")
	
	def test_GOHMM_Set(var):
		set1 = m.generateSet(50,10)
		var.assertEqual(set1.type,2)
		set2 = m.generateSet(50,1/4,"geo",6)
		set1.addSet(set2)
		var.assertEqual(set1.type,2)
		set1.save("test_save.txt")
		set2 = loadSet("test_save.txt")
		var.assertEqual(set2.type,2)
		var.assertTrue(set1.isEqual(set2))
		remove("test_save.txt")
	
	def test_BW_GOHMM(var):
		initial_model   = loadGOHMM("jajapy/tests/materials/gohmm/random_GOHMM.txt")
		training_set    = loadSet("jajapy/tests/materials/gohmm/training_set_GOHMM.txt")
		output_expected = loadGOHMM("jajapy/tests/materials/gohmm/output_GOHMM.txt")
		output_gotten   = BW_GOHMM().fit( training_set, initial_model)
		test_set = m.generateSet(10000,10)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))

if __name__ == "__main__":
	unittest.main()