import unittest
from ..ctmc import *
from os import remove
from ..base.Set import *
from math import exp

def modelCTMC1():
	labeling = ['red','red','yellow','blue','blue']
	transitions = [(0,1,0.08),(0,2,0.12),(1,1,0.3),(1,2,0.7),
				   (2,0,0.2),(2,3,0.1),(2,4,0.2),(3,3,0.8),
				   (3,1,0.1),(3,4,0.1),(4,2,0.25)]

	return createCTMC(transitions,labeling,0,"My_CTMC")

m1 = modelCTMC1()

class CTMCTestclass(unittest.TestCase):

	def test_CTMC_state(var):
		var.assertEqual(m1.e(0),1/5)
		var.assertEqual(m1.expected_time(0),5.0)
		var.assertEqual(m1.l(0,2),0.12)
		var.assertEqual(m1.l(0,3),0.0)
		var.assertEqual(m1.tau(0,2,'red'),0.6)
		var.assertEqual(m1.tau(0,2,'yellow'),0.0)
		t = 1.0
		lkl = 0.2*exp(-0.2*t)
		var.assertEqual(m1.lkl(0,t),lkl)
		var.assertEqual(m1.lkl(0,-t),0.0)
		var.assertEqual(set(m1.getAlphabet()),
						set(['red','yellow','blue']))
		var.assertEqual(m1.getLabel(2),'yellow')
	
	def test_CTMC_save_load_str(var):
		m1.save("test_save.txt")
		mprime = loadCTMC("test_save.txt")
		var.assertEqual(str(m1),str(mprime))
		remove("test_save.txt")
	
	def test_CTMC_Set(var):
		set1 = m1.generateSet(50,10,timed=True)
		var.assertEqual(set1.type,4)
		set2 = m1.generateSet(50,1/4,"geo",6, timed=True)
		set1.addSet(set2)
		var.assertEqual(set1.type,4)
		set1.save("test_save.txt")
		set2 = loadSet("test_save.txt")
		var.assertEqual(set2.type,4)
		var.assertTrue(set1.isEqual(set2))
		remove("test_save.txt")
	
	def test_CTMC_logLikelihood(var):
		set1 = Set([['red','red','red','yellow','blue']],[1])
		l11 = m1._logLikelihood_oneproc(set1)
		l12 = m1._logLikelihood_multiproc(set1)
		var.assertAlmostEqual(l11,l12)
		var.assertAlmostEqual(l11,log(0.08*5*0.3*0.7*(0.2+0.1)*2))

		set2 = Set([['red', 0.9882294310276254, 'red', 0.030426756030141687,
					 'yellow', 7.774379967385992, 'red', 13.783896165945656,
					 'yellow', 0.09826006650405217, 'red'],
					['red', 3.5876459831883487, 'yellow', 0.21449179923772158,
					 'red', 1.2177338376867959, 'red', 0.4591551146146921,
					 'yellow', 1.25461684148389, 'red']],[1,1])
		l21 = m1.logLikelihood(set2)
		var.assertAlmostEqual(l21,-12.066374414515508)
	
	def test_BW_CTMC(var):
		initial_model   = loadCTMC("jajapy/tests/materials/ctmc/random_CTMC.txt")
		
		training_set    = loadSet("jajapy/tests/materials/ctmc/training_set_timed_CTMC.txt")
		output_expected = loadCTMC("jajapy/tests/materials/ctmc/output_timed_CTMC.txt")
		output_gotten   = BW_CTMC().fit( training_set, initial_model, stormpy_output=False)
		test_set = m1.generateSet(10000,10)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))
		
		training_set    = loadSet("jajapy/tests/materials/ctmc/training_set_nontimed_CTMC.txt")
		output_expected = loadCTMC("jajapy/tests/materials/ctmc/output_nontimed_CTMC.txt")
		output_gotten   = BW_CTMC().fit( training_set, initial_model, stormpy_output=False)
		test_set = m1.generateSet(10000,10)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))
	

if __name__ == "__main__":
	unittest.main()