import unittest
from ..mc import *
from os import remove
from ..base.Set import *
from math import log
from numpy import array

def modelMC_REBER():
	alphabet = list("BTPSXVE")
	initial_state = 0
	nb_states = 7
	s0 = MC_state([(1,'B',1.0)],alphabet,nb_states)
	s1 = MC_state([(2,'T',0.5),(3,'P',0.5)],alphabet,nb_states)
	s2 = MC_state([(2,'S',0.6),(4,'X',0.4)],alphabet,nb_states)
	s3 = MC_state([(3,'T',0.7),(5,'V',0.3)],alphabet,nb_states)
	s4 = MC_state([(3,'X',0.5),(6,'S',0.5)],alphabet,nb_states)
	s5 = MC_state([(4,'P',0.5),(6,'V',0.5)],alphabet,nb_states)
	s6 = MC_state([(6,'E',1.0)],alphabet,nb_states)
	matrix = array([s0,s1,s2,s3,s4,s5,s6])
	return MC(matrix,alphabet,initial_state,"MC_REBER")

m = modelMC_REBER()

class MCTestclass(unittest.TestCase):

	def test_MC_state(var):
		var.assertEqual(m.tau(1,0,'B'),0.0)
		var.assertEqual(m.tau(1,2,'T'),0.5)
		var.assertEqual(m.tau(2,4,'X'),0.4)
		var.assertEqual(m.tau(2,4,'something else'),0.0)
		var.assertEqual(set(m.getAlphabet(1)),
						set(['T','P']))
	
	def test_MC_save_load_str(var):
		m.save("test_save.txt")
		mprime = loadMC("test_save.txt")
		mprime.name
		var.assertEqual(str(m),str(mprime))
		remove("test_save.txt")
	
	def test_MC_getAlphabet(var):
		var.assertEqual(set(m.getAlphabet()),
						set(list("BTPXSVE")))
	
	def test_MC_Set(var):
		set1 = m.generateSet(50,10)
		set2 = m.generateSet(50,1/4,"geo",6)
		set1.addSet(set2)
		var.assertEqual(set1.type,0)
		set1.save("test_save.txt")
		set2 = loadSet("test_save.txt")
		var.assertTrue(set1.isEqual(set2))
		remove("test_save.txt")

	def test_MC_logLikelihood(var):
		set1 = Set([['B','T','X','X']],[1])
		set2 = Set([['B','P','T','T']],[2])
		l11 = m._logLikelihood_oneproc(set1)
		l12 = m._logLikelihood_multiproc(set1)
		var.assertAlmostEqual(l11,l12)
		var.assertAlmostEqual(l11,log(0.1))
		l2 = m.logLikelihood(set2)
		var.assertAlmostEqual(l2,log(0.245))
		set1.addSet(set2)
		l3 = m.logLikelihood(set1)
		var.assertAlmostEqual(l3,(log(0.1)+2*log(0.245))/3)
	
	def test_BW_MC(var):
		initial_model   = loadMC("jajapy/tests/materials/mc/random_MC.txt")
		training_set    = loadSet("jajapy/tests/materials/mc/training_set_MC.txt")
		output_expected = loadMC("jajapy/tests/materials/mc/output_MC.txt")
		output_gotten   = BW_MC().fit( training_set, initial_model)
		test_set = m.generateSet(10000,10)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))
	
	def test_Alergia(var):
		training_set    = loadSet("jajapy/tests/materials/mc/training_set_MC.txt")
		Alergia().fit(training_set,0.000005)


if __name__ == "__main__":
	unittest.main()