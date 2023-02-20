import unittest
from ..hmm import *
from ..mc import *
from ..ctmc import *
from ..pctmc import *
from ..mdp import *
from ..with_stormpy import jajapyModeltoStormpy, stormpyModeltoJajapy, loadPrism
import stormpy
from os import remove

def modelMDP(p=0.75):
	labelling = ['R','L','R','L','OK','HIT']
	transitions = [(0,'m',1,p),(0,'m',2,1-p),(0,'s',3,p),(0,'s',0,1-p),
				   (1,'m',0,p),(1,'m',3,1-p),(1,'s',2,p),(1,'s',1,1-p),
				   (2,'m',5,1.0),(2,'s',4,1.0),(3,'m',5,1.0),(3,'s',4,1.0),
				   (4,'m',4,1.0),(4,'s',4,1.0),(5,'m',5,1.0),(5,'s',5,1.0)]
	return createMDP(transitions,labelling,0,"bigstreet")

def modelMC():
	labelling = list("BTSXSPTXPVVE")
	initial_state = 0
	name = "MC_REBER"
	transitions = [(0,1,0.5),(0,5,0.5),(1,2,0.6),(1,3,0.4),(2,2,0.6),(2,3,0.4),
				   (3,7,0.5),(3,4,0.5),(4,11,1.0),(5,6,0.7),(5,9,0.3),
				   (6,6,0.7),(6,9,0.3),(7,6,0.7),(7,9,0.3),(8,7,0.5),(8,4,0.5),
				   (9,8,0.5),(9,10,0.5),(10,11,1.0),(11,11,1.0)]
	return createMC(transitions,labelling,initial_state,name)

def modelCTMC():
	labelling = ['red','red','yellow','blue','blue']
	transitions = [(0,1,0.08),(0,2,0.12),(1,1,0.3),(1,2,0.7),
				   (2,0,0.2),(2,3,0.1),(2,4,0.2),(3,3,0.8),
				   (3,1,0.1),(3,4,0.1),(4,2,0.25)]
	return createCTMC(transitions,labelling,0,"My_CTMC")

def modelPCTMC():
	labelling = ['c_red_p_red','c_green_p_red','c_orange_p_green','c_red_p_green','c_orange_p_red']
	transitions = [(0,1,5.0),(1,2,'p*p/2'),(2,3,2.0),(2,4,0.1),(3,0,0.1),(4,0,2.0)]
	parameters = ['p']
	return createPCTMC(transitions,labelling,parameters,0)


class ModelConverterTestclass(unittest.TestCase):

	def test_MC_Stormpy(var):
		m = modelMC()
		mstorm = jajapyModeltoStormpy(m)
		properties = stormpy.parse_properties('P=? [F "E"]')
		result = stormpy.check_model_sparse(mstorm,properties[0])
		var.assertAlmostEqual(result.at(mstorm.initial_states[0]),1.0)

		properties = stormpy.parse_properties('P=? [G !"S"]')
		result = stormpy.check_model_sparse(mstorm,properties[0])
		var.assertAlmostEqual(result.at(mstorm.initial_states[0]),0.4)

		mprime = stormpyModeltoJajapy(mstorm)
		mprime.name = m.name
		var.assertEqual(str(mprime),str(m))
	
	def test_MC_Prism(var):
		m = modelMC()
		ts = m.generateSet(1000,15)
		mprime = loadPrism("jajapy/tests/materials/mc/reber.sm")
		var.assertAlmostEqual(m.logLikelihood(ts),mprime.logLikelihood(ts))

		m.savePrism("test_save.sm")
		mprime = loadPrism("test_save.sm")
		var.assertAlmostEqual(m.logLikelihood(ts),mprime.logLikelihood(ts))
		remove("test_save.sm")
	
	def test_MDP_Stormpy(var):
		m = modelMDP()
		actions = m.actions
		#actions.reverse()
		mstorm = jajapyModeltoStormpy(m)
		mprime = stormpyModeltoJajapy(mstorm,actions)
		mprime.name = m.name
		var.assertEqual(str(mprime),str(m))
	
	def test_MDP_Prism(var):
		m = modelMDP()
		ts = m.generateSet(1000,15,scheduler=UniformScheduler(m.actions))
		mprime = loadPrism("jajapy/tests/materials/mdp/bigstreet.sm")
		mprime.actions = m.actions #sometimes we need to reverse it
		var.assertEqual(m.logLikelihood(ts),mprime.logLikelihood(ts))

		m.savePrism("test_save.sm")
		mprime = loadPrism("test_save.sm")
		mprime.actions = m.actions #sometimes we need to reverse it
		var.assertEqual(m.logLikelihood(ts),mprime.logLikelihood(ts))
		remove("test_save.sm")

	def test_CTMC_Stormpy(var):
		m = modelCTMC()
		ts = m.generateSet(1000,15,timed=True)
		mprime = jajapyModeltoStormpy(m)
		mprime = stormpyModeltoJajapy(mprime)
		var.assertEqual(m.logLikelihood(ts),mprime.logLikelihood(ts))
	
	def test_CTMC_Prism(var):
		m = modelCTMC()
		ts = m.generateSet(1000,15,timed=True)
		mprime = loadPrism("jajapy/tests/materials/ctmc/model_ctmc_prism.sm")
		var.assertEqual(m.logLikelihood(ts),mprime.logLikelihood(ts))

		m.savePrism("test_save.sm")
		mprime = loadPrism("test_save.sm")
		var.assertEqual(m.logLikelihood(ts),mprime.logLikelihood(ts))
		remove("test_save.sm")
	
	def test_PCTMC_Stormpy(var):
		m = modelPCTMC()
		m.instantiate(['p'],[2.0])
		mprime = jajapyModeltoStormpy(m)
		mprime = stormpyModeltoJajapy(mprime)
		ts = m.generateSet(1000,15,timed=True)
		var.assertEqual(m.logLikelihood(ts),mprime.logLikelihood(ts))
	
	def test_PCTMC_Prism(var):
		m = modelPCTMC()
		m.savePrism("test_save.sm")
		m.instantiate(['p'],[2.0])
		ts = m.generateSet(1000,15,timed=True)
		mprime = loadPrism("jajapy/tests/materials/pctmc/tl.pm")
		mprime.instantiate(['p'],[2.0])
		var.assertEqual(m.logLikelihood(ts),mprime.logLikelihood(ts))
		
		mprime = loadPrism("test_save.sm")
		mprime.instantiate(['p'],[2.0])
		var.assertEqual(m.logLikelihood(ts),mprime.logLikelihood(ts))
		remove("test_save.sm")


if __name__ == "__main__":
	unittest.main()