import unittest
from ..gohmm import *
from os import remove
from ..base.Set import *
from math import log, exp, sqrt, pi

def modelGOHMM():
	s0 = GOHMM_state([[0.9,0.1],[0,1]],[3.0,5.0],0)
	s1 = GOHMM_state([[0.05,0.9,0.04,0.01],[0,1,2,4]],[0.5,1.5],1)
	s2 = GOHMM_state([[0.05,0.8,0.14,0.01],[1,2,3,4]],[0.2,0.7],2)
	s3 = GOHMM_state([[0.05,0.95],[2,3]],[0.0,0.3],3)
	s4 = GOHMM_state([[0.1,0.9],[1,4]],[2.0,4.0],4)
	return GOHMM([s0,s1,s2,s3,s4],[0.1,0.7,0.0,0.0,0.2],"GOHMM")

m = modelGOHMM()

class GOHMMTestclass(unittest.TestCase):

	def test_GOHMM_state(var):
		s0 = m.states[0]
		mu = s0.mu()
		b = exp(-0.02)/(5.0*sqrt(2*pi))
		sigma = s0.output_parameters[1]
		var.assertEqual(s0.a(0),0.9)
		var.assertEqual(s0.a(2),0.0)
		var.assertEqual(mu,3.0)
		var.assertEqual(sigma,5.0)
		var.assertEqual(s0.b(4.0),b)
		var.assertEqual(s0.tau(0,4.0),0.9*b)
	
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