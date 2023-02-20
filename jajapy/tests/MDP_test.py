import unittest
from ..mdp import *
from os import remove
from ..base.Set import *
from ..base.BW import BW
from math import log
from ..mdp.Active_BW_MDP import Active_BW_MDP
from numpy import where

def modelMDP_bigstreet(p=0.75):
	labelling = ['R','L','R','L','OK','HIT']
	transitions = [(0,'m',1,p),(0,'m',2,1-p),(0,'s',3,p),(0,'s',0,1-p),
				   (1,'m',0,p),(1,'m',3,1-p),(1,'s',2,p),(1,'s',1,1-p),
				   (2,'m',5,1.0),(2,'s',4,1.0),(3,'m',5,1.0),(3,'s',4,1.0),
				   (4,'m',4,1.0),(4,'s',4,1.0),(5,'m',5,1.0),(5,'s',5,1.0)]
	return createMDP(transitions,labelling,0,"bigstreet")

m = modelMDP_bigstreet()
scheduler = UniformScheduler(m.getActions())

class MDPTestclass(unittest.TestCase):
	
	def test_MDP_initial_state(var):
		p = 0.75
		labelling = ['R','L','R','L','OK','HIT']
		transitions = [(0,'m',1,p),(0,'m',2,1-p),(0,'s',3,p),(0,'s',0,1-p),
				   (1,'m',0,p),(1,'m',3,1-p),(1,'s',2,p),(1,'s',1,1-p),
				   (2,'m',5,1.0),(2,'s',4,1.0),(3,'m',5,1.0),(3,'s',4,1.0),
				   (4,'m',4,1.0),(4,'s',4,1.0),(5,'m',5,1.0),(5,'s',5,1.0)]
		mdp = createMDP(transitions,labelling,0)
		var.assertEqual(mdp.nb_states,7)
		var.assertEqual(mdp.labelling.count('init'),1)
		var.assertEqual(mdp.getLabel(int(where(mdp.initial_state == 1.0)[0][0])),'init')
		
		labelling = ['R','L','R','L','OK','HIT']
		mdp = createMDP(transitions,labelling,[0.3,0.0,0.0,0.2,0.5,0.0])
		var.assertEqual(mdp.nb_states,7)
		var.assertEqual(mdp.labelling.count('init'),1)
		var.assertEqual(mdp.pi(6),1.0)
		var.assertTrue((mdp.matrix[-1][0]==array([0.3,0.0,0.0,0.2,0.5,0.0,0.0])).all())
		
		labelling = ['R','L','R','L','OK','HIT']
		mdp = createMDP(transitions,labelling,array([0.3,0.0,0.0,0.2,0.5,0.0]))
		var.assertEqual(mdp.nb_states,7)
		var.assertEqual(mdp.labelling.count('init'),1)
		var.assertEqual(mdp.pi(6),1.0)
		var.assertTrue((mdp.matrix[-1][0]==array([0.3,0.0,0.0,0.2,0.5,0.0,0.0])).all())
		

	def test_MDP_state(var):
		var.assertEqual(m.tau(0,'m',1,'B'),0.0)
		var.assertEqual(m.tau(0,'m',1,'R'),0.75)
		var.assertEqual(m.tau(0,'something else',1,'R'),0.0)
		var.assertEqual(m.tau(0,'m',1,'something else'),0.0)
		var.assertEqual(set(m.getLabel(0)),
						set('R'))
		var.assertEqual(set(m.getActions(0)),
						set(['m','s']))
	
	def test_MDP_save_load_str(var):
		m.save("test_save.txt")
		mprime = loadMDP("test_save.txt")
		var.assertEqual(str(m),str(mprime))
		remove("test_save.txt")
	
	def test_MDP_observations_actions(var):
		var.assertEqual(set(m.getAlphabet()),
						set(['L','R','HIT','OK','init']))
		var.assertEqual(set(m.getActions()),
						set(['m','s']))
	
	def test_MDP_Set(var):
		set1 = m.generateSet(50,10,scheduler)
		set2 = m.generateSet(50,1/4,scheduler,"geo",6)
		set1.addSet(set2)
		var.assertEqual(set1.type,1)
		set1.save("test_save.txt")
		set2 = loadSet("test_save.txt")
		var.assertTrue(set1.isEqual(set2))
		remove("test_save.txt")

	def test_MDP_logLikelihood(var):
		set1 = Set([['init','m','R','m','L','s','L','s','R','m','HIT','m','HIT','s','HIT']],[1],t=1)
		set2 = Set([['init','m','R','s','R','m','R','s','OK','m','OK','m','OK']],[2],t=1)
		l12 = m._logLikelihood_multiproc(set1)
		l11 = m._logLikelihood_oneproc(set1)
		var.assertAlmostEqual(l11,l12)
		var.assertAlmostEqual(l11,log(9/64))
		l2 = m.logLikelihood(set2)
		var.assertAlmostEqual(l2,log(1/16))
		set1.addSet(set2)
		l3 = m.logLikelihood(set1)
		var.assertAlmostEqual(l3,(log(9/64)+2*log(1/16))/3)
	
	def test_BW_MDP(var):
		initial_model   = loadMDP("jajapy/tests/materials/mdp/random_MDP.txt")
		training_set    = loadSet("jajapy/tests/materials/mdp/training_set_MDP.txt")
		output_expected = loadMDP("jajapy/tests/materials/mdp/output_MDP.txt")
		output_gotten   = BW().fit( training_set, initial_model, stormpy_output=False)
		test_set = m.generateSet(10000,10,scheduler)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))
	
	def test_Active_BW_MDP(var):
		initial_model   = loadMDP("jajapy/tests/materials/mdp/random_MDP.txt")
		training_set    = loadSet("jajapy/tests/materials/mdp/training_set_MDP.txt")
		output_expected = loadMDP("jajapy/tests/materials/mdp/active_output_MDP.txt")
		output_gotten   = Active_BW_MDP().fit(training_set, sul=m,initial_model=initial_model, lr=0,
											  nb_iterations=10,nb_sequences=10,sequence_length=10,
											  stormpy_output=False)
		test_set = m.generateSet(10000,10,scheduler)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set),places=2)


	
	def test_IOAlergia(var):
		training_set    = loadSet("jajapy/tests/materials/mdp/training_set_MDP.txt")
		IOAlergia().fit(training_set,0.01, stormpy_output=False)

	def test_UniformScheduler(var):
		nb_trials = 100000
		actions = m.getActions()
		results = [0.0 for i in actions]
		for i in range(nb_trials):
			results[actions.index(scheduler.getAction())] += 1/nb_trials
		for i in range(len(results)):
			var.assertAlmostEqual(results[i],len(results)/nb_trials,delta=3)


if __name__ == "__main__":
	unittest.main()