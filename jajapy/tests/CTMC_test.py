import unittest
from ..ctmc import *
from os import remove
from ..base.Set import *
from math import exp

def modelCTMC(suffix=''):
	s0 = CTMC_state([[0.3/5,0.5/5,0.2/5],[1,2,3], ['r'+suffix,'g'+suffix,'r'+suffix]],0)
	s1 = CTMC_state([[0.08,0.25,0.6,0.07],[0,2,2,3], ['r'+suffix,'r'+suffix,'g'+suffix,'b'+suffix]],1)
	s2 = CTMC_state([[0.5/4,0.2/4,0.3/4],[1,3,3], ['b'+suffix,'g'+suffix,'r'+suffix]],2)
	s3 = CTMC_state([[0.95/2,0.04/2,0.01/2],[0,0,2], ['r'+suffix,'g'+suffix,'r'+suffix]],3)
	return CTMC([s0,s1,s2,s3],0,"CTMC")

m = modelCTMC()

class CTMCTestclass(unittest.TestCase):

	def test_CTMC_state(var):
		s0 = m.states[0]
		var.assertEqual(s0.e(),1/5)
		var.assertEqual(s0.expected_time(),5.0)
		var.assertEqual(s0.l(2,'r'),0.0)
		var.assertEqual(s0.l(2,'g'),1/10)
		var.assertEqual(s0.tau(2,'g'),1/2)
		t = 1.0
		lkl = 0.2*exp(-0.2*t)
		var.assertEqual(s0.lkl(t),lkl)
		var.assertEqual(s0.lkl(-t),0.0)
		var.assertEqual(set(s0.observations()),
						set(['r','g']))
	
	def test_CTMC_save_load_str(var):
		m.save("test_save.txt")
		mprime = loadCTMC("test_save.txt")
		var.assertEqual(str(m),str(mprime))
		remove("test_save.txt")
	
	def test_CTMC_Set(var):
		set1 = m.generateSet(50,10,timed=True)
		var.assertEqual(set1.type,4)
		set2 = m.generateSet(50,1/4,"geo",6, timed=True)
		set1.addSet(set2)
		var.assertEqual(set1.type,4)
		set1.save("test_save.txt")
		set2 = loadSet("test_save.txt")
		var.assertEqual(set2.type,4)
		var.assertTrue(set1.isEqual(set2))
		remove("test_save.txt")
	
	def test_BW_CTMC(var):
		initial_model   = loadCTMC("jajapy/tests/materials/ctmc/random_CTMC.txt")
		training_set    = loadSet("jajapy/tests/materials/ctmc/training_set_CTMC.txt")
		output_expected = loadCTMC("jajapy/tests/materials/ctmc/output_CTMC.txt")
		output_gotten   = BW_CTMC().fit( training_set, initial_model)
		test_set = m.generateSet(10000,10)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))

if __name__ == "__main__":
	unittest.main()