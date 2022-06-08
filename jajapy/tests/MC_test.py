import unittest
from ..mc import *
from os import remove
from ..base.Set import *
from math import log

def modelMC_REBER():
	s0 = MC_state([[1.0],[1],['B']],0)
	s1 = MC_state([[0.5,0.5],[2,3],['T','P']],1)
	s2 = MC_state([[0.6,0.4],[2,4],['S','X']],2)
	s3 = MC_state([[0.7,0.3],[3,5],['T','V']],3)
	s4 = MC_state([[0.5,0.5],[3,6],['X','S']],4)
	s5 = MC_state([[0.5,0.5],[4,6],['P','V']],5)
	s6 = MC_state([[1.0],[6],['E']],6)
	return MC([s0,s1,s2,s3,s4,s5,s6],0,"MC_REBER")

m = modelMC_REBER()

class MCTestclass(unittest.TestCase):

	def test_MC_state(var):
		s1 = m.states[1]
		s2 = m.states[2]
		var.assertEqual(s1.tau(0,'B'),0.0)
		var.assertEqual(s1.tau(2,'T'),0.5)
		var.assertEqual(s2.tau(4,'X'),0.4)
		var.assertEqual(s2.tau(4,'something else'),0.0)
		var.assertEqual(set(s1.observations()),
						set(['T','P']))
	
	def test_MC_save_load_str(var):
		m.save("test_save.txt")
		mprime = loadMC("test_save.txt")
		var.assertEqual(str(m),str(mprime))
		remove("test_save.txt")
	
	def test_MC_observations(var):
		var.assertEqual(set(m.observations()),
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