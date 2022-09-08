import unittest
from ..hmm import *
from os import remove
from ..base.Set import *
from math import log

def modelHMM4():
	s0 = HMM_state([("x",0.4),("y",0.6)],[(1,0.5),(2,0.5)],0)
	s1 = HMM_state([("a",0.8),("b",0.2)],[(3,1.0)],1)
	s2 = HMM_state([("a",0.1),("b",0.9)],[(4,1.0)],2)
	s3 = HMM_state([("x",0.5),("y",0.5)],[(0,0.8),(1,0.1),(2,0.1)],3)
	s4 = HMM_state([("y",1.0)],[(3,1.0)],4)
	return HMM([s0,s1,s2,s3,s4],0,"HMM4")

m = modelHMM4()

class HMMTestclass(unittest.TestCase):

	def test_HMM_state(var):
		s0 = m.states[0]
		var.assertEqual(s0.a(0),0.0)
		var.assertEqual(s0.a(1),0.5)
		var.assertEqual(s0.b('x'),0.4)
		var.assertEqual(s0.b('something else'),0.0)
		var.assertEqual(s0.tau(1,'x'),0.2)
		var.assertEqual(set(s0.observations()),set(['x','y']))
	
	def test_HMM_save_load_str(var):
		m.save("test_save.txt")
		mprime = loadHMM("test_save.txt")
		var.assertEqual(str(m),str(mprime))
		remove("test_save.txt")
	
	def test_HMM_observations(var):
		var.assertEqual(set(m.observations()),set(['x','y','a','b']))
	
	def test_HMM_Set(var):
		set1 = m.generateSet(50,10)
		set2 = m.generateSet(50,1/4,"geo",6)
		set1.addSet(set2)
		var.assertEqual(set1.type,0)
		set1.save("test_save.txt")
		set2 = loadSet("test_save.txt")
		var.assertTrue(set1.isEqual(set2))
		remove("test_save.txt")

	def test_HMM_logLikelihood(var):
		set1 = Set([['x','a','x','y']],[1])
		set2 = Set([['y','b','y','x']],[2])
		l11 = m._logLikelihood_oneproc(set1)
		l12 = m._logLikelihood_multiproc(set1)
		var.assertAlmostEqual(l11,l12)
		var.assertAlmostEqual(l11,log(0.0384))
		l2 = m.logLikelihood(set2)
		var.assertAlmostEqual(l2,log(0.1446))
		set1.addSet(set2)
		l3 = m.logLikelihood(set1)
		var.assertAlmostEqual(l3,(log(0.0384)+2*log(0.1446))/3)
	
	def test_BW_HMM(var):
		initial_model   = loadHMM("jajapy/tests/materials/hmm/random_HMM.txt")
		training_set    = loadSet("jajapy/tests/materials/hmm/training_set_HMM.txt")
		output_expected = loadHMM("jajapy/tests/materials/hmm/output_HMM.txt")
		output_gotten   = BW_HMM().fit( training_set, initial_model)
		test_set = m.generateSet(10000,10)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))

if __name__ == "__main__":
	unittest.main()