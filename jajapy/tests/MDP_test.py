import unittest
from ..mdp import *
from os import remove
from ..base.Set import *
from math import log

def modelMDP_bigstreet(p=0.75):
	m_s_rr = MDP_state({'m': [[p,1-p],[1,2],['L','R']], 's': [[p,1-p],[2,0],['L','R']]},0)
	m_s_ll = MDP_state({'m': [[p,1-p],[0,2],['R','L']], 's': [[p,1-p],[2,1],['R','L']]},1)
	m_s_di = MDP_state({'m': [[1.0],[3],['HIT']],       's': [[1.0],[4],['OK']]},2)
	m_s_de = MDP_state({'m': [[1.0],[3],['HIT']],       's': [[1.0],[3],['HIT']]},3)
	m_s_vi = MDP_state({'m': [[1.0],[4],['OK']],        's': [[1.0],[4],['OK']]},4)
	return MDP([m_s_rr,m_s_ll,m_s_di,m_s_de,m_s_vi],0,"bigstreet")

m = modelMDP_bigstreet()
scheduler = UniformScheduler(m.actions())

class MDPTestclass(unittest.TestCase):

	def test_MDP_state(var):
		s0 = m.states[0]
		var.assertEqual(s0.tau('m',1,'B'),0.0)
		var.assertEqual(s0.tau('m',1,'L'),0.75)
		var.assertEqual(s0.tau('something else',1,'L'),0.0)
		var.assertEqual(s0.tau('m',1,'something else'),0.0)
		var.assertEqual(set(s0.observations()),
						set(['L','R']))
		var.assertEqual(set(s0.actions()),
						set(['m','s']))
	
	def test_MDP_save_load_str(var):
		m.save("test_save.txt")
		mprime = loadMDP("test_save.txt")
		var.assertEqual(str(m),str(mprime))
		remove("test_save.txt")
	
	def test_MDP_observations_actions(var):
		var.assertEqual(set(m.observations()),
						set(['L','R','HIT','OK']))
		var.assertEqual(set(m.actions()),
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
		set1 = Set([['m','L','s','L','s','R','m','HIT','m','HIT','s','HIT']],[1],from_MDP=True)
		set2 = Set([['s','R','m','R','s','OK','m','OK','m','OK']],[2],from_MDP=True)
		l11 = m._logLikelihood_oneproc(set1)
		l12 = m._logLikelihood_multiproc(set1)
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
		output_gotten   = BW_MDP().fit( training_set, initial_model)
		test_set = m.generateSet(10000,10,scheduler)
		var.assertAlmostEqual(output_expected.logLikelihood(test_set),
							  output_gotten.logLikelihood(test_set))
	
	def test_IOAlergia(var):
		training_set    = loadSet("jajapy/tests/materials/mdp/training_set_MDP.txt")
		IOAlergia().fit(training_set,0.0005)

	def test_UniformScheduler(var):
		nb_trials = 100000
		actions = m.actions()
		results = [0.0 for i in actions]
		for i in range(nb_trials):
			results[actions.index(scheduler.getAction())] += 1/nb_trials
		for i in range(len(results)):
			var.assertAlmostEqual(results[i],len(results)/nb_trials,delta=3)


if __name__ == "__main__":
	unittest.main()