import unittest
from ..mc import *
from os import remove
from ..base.Set import *
from ..base.BW import BW
from math import log
from numpy import where, array

def modelMC_REBER():
	labelling = list("BTSXSPTXPVVE")
	initial_state = 0
	name = "MC_REBER"
	transitions = [(0,1,0.5),(0,5,0.5),(1,2,0.6),(1,3,0.4),(2,2,0.6),(2,3,0.4),
				   (3,7,0.5),(3,4,0.5),(4,11,1.0),(5,6,0.7),(5,9,0.3),
				   (6,6,0.7),(6,9,0.3),(7,6,0.7),(7,9,0.3),(8,7,0.5),(8,4,0.5),
				   (9,8,0.5),(9,10,0.5),(10,11,1.0),(11,11,1.0)]
	return createMC(transitions,labelling,initial_state,name)

m = modelMC_REBER()

class MCTestclass(unittest.TestCase):

	def test_MC_initial_state(var):
		labelling=['a','b','c','d','a']
		transitions = [(0,1,0.8),(0,2,0.2),
				   (1,3,0.6),(1,2,0.4),
				   (2,0,0.5),(2,4,0.5),
				   (3,2,0.3),(3,3,0.7),
				   (4,2,0.2),(4,3,0.1),(4,4,0.7)]
		mc = createMC(transitions,labelling,0)
		var.assertEqual(mc.nb_states,6)
		var.assertEqual(mc.labelling.count('init'),1)
		var.assertEqual(mc.getLabel(int(where(mc.initial_state == 1.0)[0][0])),'init')
		
		labelling=['a','b','c','d','a']
		mc = createMC(transitions,labelling,[0.3,0.0,0.0,0.2,0.5])
		var.assertEqual(mc.nb_states,6)
		var.assertEqual(mc.labelling.count('init'),1)
		var.assertEqual(mc.pi(5),1.0)
		var.assertTrue((mc.matrix[-1]==array([0.3,0.0,0.0,0.2,0.5,0.0])).all())
		
		labelling=['a','b','c','d','a']
		mc = createMC(transitions,labelling,array([0.3,0.0,0.0,0.2,0.5]))
		var.assertEqual(mc.nb_states,6)
		var.assertEqual(mc.labelling.count('init'),1)
		var.assertEqual(mc.pi(5),1.0)
		var.assertTrue((mc.matrix[-1]==array([0.3,0.0,0.0,0.2,0.5,0.0])).all())
		

	def test_MC_state(var):
		var.assertEqual(m.tau(1,0,'B'),0.0)
		var.assertEqual(m.tau(0,1,'B'),0.5)
		var.assertEqual(m.getLabel(2),'S')
		var.assertEqual(m.tau(2,3,'S'),0.4)
		var.assertEqual(m.tau(2,3,'something else'),0.0)
		var.assertEqual(m.getLabel(1),'T')

	
	def test_MC_save_load_str(var):
		m.save("test_save.txt")
		mprime = loadMC("test_save.txt")
		var.assertEqual(str(m),str(mprime))
		remove("test_save.txt")
	
	def test_MC_getAlphabet(var):
		var.assertEqual(set(m.getAlphabet()),
						set(list("BTPXSVE")+['init']))
	
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
		set1 = Set([['init','B','T','X','X']],[1])
		set2 = Set([['init','B','P','T','T']],[2])
		l11 = m._logLikelihood_oneproc(set1)
		l12 = m._logLikelihood_multiproc(set1)
		var.assertAlmostEqual(l11,l12)
		var.assertAlmostEqual(l11,log(0.1))
		l2 = m.logLikelihood(set2)
		var.assertAlmostEqual(l2,log(0.245))
		set1.addSet(set2)
		l3 = m.logLikelihood(set1)
		var.assertAlmostEqual(l3,(log(0.1)+2*log(0.245))/3)
	
	def test_MC_random(var):
		alphabet = list("BTSXPVE")
		random_model = MC_random(11, alphabet, False)
		for i in alphabet:
			var.assertGreaterEqual(random_model.labelling.count(i),1)

	def test_BW_MC(var):
		initial_model   = loadMC("jajapy/tests/materials/mc/random_MC.txt")
		training_set    = loadSet("jajapy/tests/materials/mc/training_set_MC.txt")
		output_expected = loadMC("jajapy/tests/materials/mc/output_MC.txt")
		output_gotten   = BW().fit( training_set, initial_model, stormpy_output=False)
		test_set = m.generateSet(10000,10)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))
	
	def test_Alergia(var):
		training_set    = loadSet("jajapy/tests/materials/mc/training_set_MC.txt")
		output_expected = loadMC("jajapy/tests/materials/mc/output_alergia_MC.txt")
		output_gotten =  Alergia().fit(training_set,0.000005,stormpy_output=False)
		test_set = m.generateSet(10000,10)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))


if __name__ == "__main__":
	unittest.main()