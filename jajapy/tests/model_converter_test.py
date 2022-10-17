import unittest
from ..hmm import *
from ..mc import *
from ..ctmc import *
from ..mdp import *
from ..with_stormpy import jajapyModeltoStorm, stormModeltoJajapy
import stormpy

def modelHMM():
	alphabet = list("abwxyz")
	nb_states = 6
	s0 = HMM_state([("x",0.4),("y",0.6)],[(1,0.5),(2,0.5)],alphabet,nb_states)
	s1 = HMM_state([("a",0.8),("b",0.2)],[(3,1.0)],alphabet,nb_states)
	s2 = HMM_state([("a",0.1),("b",0.9)],[(4,0.4),(5,0.6)],alphabet,nb_states)
	s3 = HMM_state([("x",0.5),("y",0.5)],[(0,0.8),(1,0.1),(2,0.1)],alphabet,nb_states)
	s4 = HMM_state([("z",1.0)],[(4,1.0)],alphabet,nb_states)
	s5 = HMM_state([("w",1.0)],[(5,1.0)],alphabet,nb_states)
	transitions = array([s0[0],s1[0],s2[0],s3[0],s4[0],s5[0]])
	output 	    = array([s0[1],s1[1],s2[1],s3[1],s4[1],s5[1]])
	return HMM(transitions,output,alphabet,0,"HMM")

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

def modelMDP_bigstreet(p=0.75):
	alphabet = ['L','R','OK','HIT']
	actions  = ['m','s']
	nb_states = 5
	m_s_rr = MDP_state({'m': list(zip([1,2],['L','R'],[p,1-p])), 's': list(zip([2,0],['L','R'],[p,1-p]))},alphabet,nb_states,actions)
	m_s_ll = MDP_state({'m': list(zip([0,2],['R','L'],[p,1-p])), 's': list(zip([2,1],['R','L'],[p,1-p]))},alphabet,nb_states,actions)
	m_s_di = MDP_state({'m': [(3,'HIT',1.0)],       's': [(4,'OK',1.0)]},alphabet,nb_states,actions)
	m_s_de = MDP_state({'m': [(3,'HIT',1.0)],       's': [(3,'HIT',1.0)]},alphabet,nb_states,actions)
	m_s_vi = MDP_state({'m': [(4,'OK' ,1.0)],       's': [(4,'OK',1.0)]},alphabet,nb_states,actions)
	matrix = array([m_s_rr,m_s_ll,m_s_di,m_s_de,m_s_vi])
	return MDP(matrix,alphabet,actions,0,"bigstreet")

def modelCTMC():
	alphabet = ['r', 'g', 'b']
	nb_states = 4
	s0 = CTMC_state(list(zip([1,2,3], ['r','g','r'],[0.3/5,0.5/5,0.2/5])),alphabet,nb_states)
	s1 = CTMC_state(list(zip([0,2,2,3], ['r','r','g','b'],[0.08,0.25,0.6,0.07])),alphabet,nb_states)
	s2 = CTMC_state(list(zip([1,3,3], ['b','g','r'],[0.5/4,0.2/4,0.3/4])),alphabet,nb_states)
	s3 = CTMC_state(list(zip([0,0,2], ['r','g','r'],[0.95/2,0.04/2,0.01/2])),alphabet,nb_states)
	return CTMC(array([s0,s1,s2,s3]),alphabet,0,"CTMC1")


class ModelConverterTestclass(unittest.TestCase):

	def test_HMM(var):
		m = modelHMM()

		ts = m.generateSet(1000,10)
		ll1 = m.logLikelihood(ts)

		m = jajapyModeltoStorm(m)
		properties = stormpy.parse_properties('P=? [F "w"]')
		result = stormpy.check_model_sparse(m,properties[0])
		var.assertAlmostEqual(result.at(m.initial_states[0]),0.6)

		properties = stormpy.parse_properties('P=? [F "z"]')
		result = stormpy.check_model_sparse(m,properties[0])
		var.assertAlmostEqual(result.at(m.initial_states[0]),0.4)

		m = stormModeltoJajapy(m)
		ll2 = m.logLikelihood(ts)
		var.assertAlmostEqual(ll1,ll2)
	
	def test_MC(var):
		m = modelMC_REBER()

		ts = m.generateSet(1000,10)
		ll1 = m.logLikelihood(ts)

		m = jajapyModeltoStorm(m)
		properties = stormpy.parse_properties('P=? [F "E"]')
		result = stormpy.check_model_sparse(m,properties[0])
		var.assertAlmostEqual(result.at(m.initial_states[0]),1.0)

		properties = stormpy.parse_properties('P=? [G !"S"]')
		result = stormpy.check_model_sparse(m,properties[0])
		var.assertAlmostEqual(result.at(m.initial_states[0]),0.4)

		m = stormModeltoJajapy(m)
		ll2 = m.logLikelihood(ts)
		var.assertAlmostEqual(ll1,ll2)
	
	def test_MDP(var):
		m = modelMDP_bigstreet()
		actions = m.actions

		ts = m.generateSet(1000,10,scheduler=UniformScheduler(actions))
		ll1 = m.logLikelihood(ts)

		m = jajapyModeltoStorm(m)
		properties = stormpy.parse_properties('Pmax=? [F "OK" ]')
		result = stormpy.check_model_sparse(m,properties[0])
		var.assertAlmostEqual(result.at(m.initial_states[0]),1.0,places=5)

		properties = stormpy.parse_properties('Rmax=? [F "OK" | "HIT" ]')
		result = stormpy.check_model_sparse(m,properties[0])
		var.assertAlmostEqual(result.at(m.initial_states[0]),-7/3,places=5)

		m2 = stormModeltoJajapy(m,actions)
		ll2 = m2.logLikelihood(ts)
		if abs(ll2-ll1) > 0.1:
			m2.actions.reverse()
			ll2 = m2.logLikelihood(ts)
			
		var.assertAlmostEqual(ll1,ll2)

	def test_CTMC(var):
		m = modelCTMC()

		ts = m.generateSet(1000,10,timed=True)
		ll1 = m.logLikelihood(ts)

		m = jajapyModeltoStorm(m)
		properties = stormpy.parse_properties('T=? [F "b" ]')
		result = stormpy.check_model_sparse(m,properties[0])
		var.assertAlmostEqual(result.at(m.initial_states[0]),5.0,places=5)

		m = stormModeltoJajapy(m)
		ll2 = m.logLikelihood(ts)
		var.assertAlmostEqual(ll1,ll2)


if __name__ == "__main__":
	unittest.main()