import unittest
from ..ctmc import *
from os import remove
from ..base.Set import *
from math import exp

def modelCTMC1(suffix=''):
	alphabet = ['r'+suffix, 'g'+suffix, 'b'+suffix]
	nb_states = 4
	s0 = CTMC_state(list(zip([1,2,3], ['r'+suffix,'g'+suffix,'r'+suffix],[0.3/5,0.5/5,0.2/5])),alphabet,nb_states)
	s1 = CTMC_state(list(zip([0,2,2,3], ['r'+suffix,'r'+suffix,'g'+suffix,'b'+suffix],[0.08,0.25,0.6,0.07])),alphabet,nb_states)
	s2 = CTMC_state(list(zip([1,3,3], ['b'+suffix,'g'+suffix,'r'+suffix],[0.5/4,0.2/4,0.3/4])),alphabet,nb_states)
	s3 = CTMC_state(list(zip([0,0,2], ['r'+suffix,'g'+suffix,'r'+suffix],[0.95/2,0.04/2,0.01/2])),alphabet,nb_states)
	return CTMC(array([s0,s1,s2,s3]),alphabet,0,"CTMC1")

def modelCTMC2(suffix=''):
	alphabet = ['r'+suffix, 'g'+suffix, 'b'+suffix]
	nb_states = 2
	s0 = CTMC_state(list(zip([1,1], ['r'+suffix,'g'+suffix],[0.6/3,0.4/3])),alphabet,nb_states)
	s1 = CTMC_state(list(zip([0,0,0], ['r'+suffix,'g'+suffix,'b'+suffix],[0.2/4,0.7/4,0.1/4])),alphabet,nb_states)
	return CTMC(array([s0,s1]),alphabet,0,"CTMC2")

def modelAsynchronous(disjoint):
	if disjoint:
		r1, g1, b1, r2, g2, b2 = 'r1', 'g1', 'b1', 'r2', 'g2', 'b2'
	else:
		r1, g1, b1, r2, g2, b2 = 'r', 'g', 'b', 'r', 'g', 'b'
	alphabet = [r1, g1, b1, r2, g2, b2]
	nb_states = 8
	s00 = CTMC_state(list(zip([2,4,6,1,1],[r1,g1,r1,r2,g2],[0.3/5,0.5/5,0.2/5,0.6/3,0.4/3])),alphabet,nb_states)
	s01 = CTMC_state(list(zip([3,5,7,0,0,0],[r1,g1,r1,r2,g2,b2],[0.3/5,0.5/5,0.2/5,0.2/4,0.7/4,0.1/4])),alphabet,nb_states)
	s10 = CTMC_state(list(zip([0,4,4,6,3,3],[r1,r1,g1,b1,r2,g2],[0.08,0.25,0.6,0.07,0.6/3,0.4/3])),alphabet,nb_states)
	s11 = CTMC_state(list(zip([1,5,5,7,2,2,2],[r1,r1,g1,b1,r2,g2,b2],[0.08,0.25,0.6,0.07,0.2/4,0.7/4,0.1/4])),alphabet,nb_states)
	s20 = CTMC_state(list(zip([2,6,6,5,5],[b1,g1,r1,r2,g2],[0.5/4,0.2/4,0.3/4,0.6/3,0.4/3])),alphabet,nb_states)
	s21 = CTMC_state(list(zip([3,7,7,4,4,4],[b1,g1,r1,r2,g2,b2],[0.5/4,0.2/4,0.3/4,0.2/4,0.7/4,0.1/4])),alphabet,nb_states)
	s30 = CTMC_state(list(zip([0,0,4,7,7],[r1,g1,r1,r2,g2],[0.95/2,0.04/2,0.01/2,0.6/3,0.4/3])),alphabet,nb_states)
	s31 = CTMC_state(list(zip([1,1,5,6,6,6],[r1,g1,r1,r2,g2,b2],[0.95/2,0.04/2,0.01/2,0.2/4,0.7/4,0.1/4])),alphabet,nb_states)
	return CTMC(array([s00,s01,s10,s11,s20,s21,s30,s31]),alphabet,0,"compo")

m1 = modelCTMC1()
m2 = modelCTMC2()

class CTMCTestclass(unittest.TestCase):

	def test_CTMC_state(var):
		var.assertEqual(m1.e(0),1/5)
		var.assertEqual(m1.expected_time(0),5.0)
		var.assertEqual(m1.l(0,2,'r'),0.0)
		var.assertEqual(m1.l(0,2,'g'),1/10)
		var.assertEqual(m1.tau(0,2,'g'),1/2)
		t = 1.0
		lkl = 0.2*exp(-0.2*t)
		var.assertEqual(m1.lkl(0,t),lkl)
		var.assertEqual(m1.lkl(0,-t),0.0)
		var.assertEqual(set(m1.getAlphabet(0)),
						set(['r','g']))
		var.assertEqual(set(m1.getAlphabet()),
						set(['r','g','b']))
	
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
	
	def test_BW_CTMC(var):
		initial_model   = loadCTMC("jajapy/tests/materials/ctmc/random_CTMC.txt")
		training_set    = loadSet("jajapy/tests/materials/ctmc/training_set_CTMC.txt")
		output_expected = loadCTMC("jajapy/tests/materials/ctmc/output_CTMC.txt")
		output_gotten   = BW_CTMC().fit( training_set, initial_model)
		test_set = m1.generateSet(10000,10)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))
	
	def test_CTMC_AsynchronousComposition(var):
		compo1 = asynchronousComposition(m1,m2,disjoint=True,name="compo")
		compo2 = modelAsynchronous(disjoint=True)
		s = compo2.generateSet(1000,10,timed=True)
		var.assertAlmostEqual(compo2.logLikelihood(s),compo1.logLikelihood(s))

		compo1 = asynchronousComposition(m1,m2,disjoint=False,name="compo")
		compo2 = modelAsynchronous(disjoint=False)
		s = compo2.generateSet(1000,10,timed=True)
		var.assertAlmostEqual(compo2.logLikelihood(s),compo1.logLikelihood(s))
	
	def test_MM_CTMC_Composition(var):
		# 2 components timed
		initial_model1 = loadCTMC("jajapy/tests/materials/ctmc/random_CTMC_1.txt")
		initial_model2 = loadCTMC("jajapy/tests/materials/ctmc/random_CTMC_2.txt")
		training_set = loadSet("jajapy/tests/materials/ctmc/training_set_CTMC_compo_timed.txt")
		output_expected_1 = loadCTMC("jajapy/tests/materials/ctmc/output_CTMC_1_timed.txt")
		output_expected_2 = loadCTMC("jajapy/tests/materials/ctmc/output_CTMC_2_timed.txt")
		output_gotten_1, output_gotten_2 = MM_CTMC_Composition().fit(training_set,
																	 initial_model_1=initial_model1,
																	 initial_model_2=initial_model2)
		
		test_set = m2.generateSet(100,10,timed=True)
		var.assertAlmostEqual(output_expected_2.logLikelihood(test_set),
							  output_gotten_2.logLikelihood(test_set))
		test_set = m1.generateSet(100,10,timed=True)
		var.assertAlmostEqual(output_expected_1.logLikelihood(test_set),
							  output_gotten_1.logLikelihood(test_set))
		
		# 1 component non-timed
		training_set = loadSet("jajapy/tests/materials/ctmc/training_set_CTMC_compo_nontimed.txt")
		output_expected_1 = loadCTMC("jajapy/tests/materials/ctmc/output_CTMC_1_nontimed.txt")
		output_gotten_1,_ = MM_CTMC_Composition().fit(training_set,
													 initial_model_1=initial_model1,
													 initial_model_2=m2,
													 to_update=1)
		var.assertAlmostEqual(output_expected_1.logLikelihood(test_set),
							  output_gotten_1.logLikelihood(test_set))
		


if __name__ == "__main__":
	unittest.main()