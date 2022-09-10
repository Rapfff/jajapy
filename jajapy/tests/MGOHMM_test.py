import unittest
from ..mgohmm import *
from os import remove
from ..base.Set import *
from math import log, exp, sqrt, pi

def modelMGOHMM():
	nb_states = 5
	s0 = MGOHMM_state([(0,0.9),(1,0.1)],[[3.0,5.0],[2.0,4.0]],nb_states)
	s1 = MGOHMM_state([(0,0.05),(1,0.9),(2,0.04),(4,0.01)],[[0.5,1.5],[2.5,1.5]],nb_states)
	s2 = MGOHMM_state([(1,0.05),(2,0.8),(3,0.14),(4,0.01)],[[0.2,0.7],[1.0,1.0]],nb_states)
	s3 = MGOHMM_state([(2,0.05),(3,0.95)],[[0.0,0.3],[1.5,5.0]],nb_states)
	s4 = MGOHMM_state([(1,0.1),(4,0.9)],[[2.0,4.0],[0.5,0.5]],nb_states)
	matrix = array([s0[0],s1[0],s2[0],s3[0],s4[0]])
	output = array([s0[1],s1[1],s2[1],s3[1],s4[1]])
	return MGOHMM(matrix,output,[0.1,0.7,0.0,0.0,0.2],name="MGOHMM")

m = modelMGOHMM()

class MGOHMMTestclass(unittest.TestCase):

	def test_MGOHMM_state(var):
		mu = m.mu(0)
		b = exp(-0.02)/(5.0*sqrt(2*pi))*exp(-1/32)/(4.0*sqrt(2*pi))
		sigma = [m.output[0][i][1] for i in range(2)]
		
		var.assertEqual(m.a(0,0),0.9)
		var.assertEqual(m.a(0,2),0.0)
		var.assertEqual(sigma,[5.0,4.0])
		var.assertTrue(all(mu==array([3.0,2.0])))
		var.assertEqual(m.b(0,[4.0,3.0]),b)
		var.assertEqual(m.tau(0,0,[4.0,3.0]),0.9*b)
	
	def test_MGOHMM_save_load_str(var):
		m.save("test_save.txt")
		mprime = loadMGOHMM("test_save.txt")
		var.assertEqual(type(m),type(mprime))
		var.assertEqual(type(m),MGOHMM)
		
		var.assertEqual(str(m),str(mprime))
		remove("test_save.txt")
	
	def test_MGOHMM_Set(var):
		set1 = m.generateSet(50,10)
		var.assertEqual(set1.type,3)
		set2 = m.generateSet(50,1/4,"geo",6)
		set1.addSet(set2)
		set1.save("test_save.txt")
		set2 = loadSet("test_save.txt")
		var.assertEqual(set2.type,3)
		var.assertTrue(set1.isEqual(set2))
		remove("test_save.txt")
	
	def test_BW_MGOHMM(var):
		initial_model   = loadMGOHMM("jajapy/tests/materials/mgohmm/random_MGOHMM.txt")
		training_set    = loadSet("jajapy/tests/materials/mgohmm/training_set_MGOHMM.txt")
		output_expected = loadMGOHMM("jajapy/tests/materials/mgohmm/output_MGOHMM.txt")
		output_gotten   = BW_MGOHMM().fit( training_set, initial_model)
		test_set = m.generateSet(10000,10)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))

if __name__ == "__main__":
	unittest.main()