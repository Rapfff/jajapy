import unittest
from ..hmm import *
from ..mc import *
from ..ctmc import *
from ..mdp import *
from ..with_stormpy import jajapyModeltoStormpy, stormpyModeltoJajapy
import stormpy

def modelMDP(p=0.75):
	labeling = ['R','L','R','L','OK','HIT']
	transitions = [(0,'m',1,p),(0,'m',2,1-p),(0,'s',3,p),(0,'s',0,1-p),
				   (1,'m',0,p),(1,'m',3,1-p),(1,'s',2,p),(1,'s',1,1-p),
				   (2,'m',5,1.0),(2,'s',4,1.0),(3,'m',5,1.0),(3,'s',4,1.0),
				   (4,'m',4,1.0),(4,'s',4,1.0),(5,'m',5,1.0),(5,'s',5,1.0)]
	return createMDP(transitions,labeling,0,"bigstreet")

def modelMC():
	labeling = list("BTSXSPTXPVVE")
	initial_state = 0
	name = "MC_REBER"
	transitions = [(0,1,0.5),(0,5,0.5),(1,2,0.6),(1,3,0.4),(2,2,0.6),(2,3,0.4),
				   (3,7,0.5),(3,4,0.5),(4,11,1.0),(5,6,0.7),(5,9,0.3),
				   (6,6,0.7),(6,9,0.3),(7,6,0.7),(7,9,0.3),(8,7,0.5),(8,4,0.5),
				   (9,8,0.5),(9,10,0.5),(10,11,1.0),(11,11,1.0)]
	return createMC(transitions,labeling,initial_state,name)

def modelCTMC():
	labeling = ['red','red','yellow','blue','blue']
	transitions = [(0,1,0.08),(0,2,0.12),(1,1,0.3),(1,2,0.7),
				   (2,0,0.2),(2,3,0.1),(2,4,0.2),(3,3,0.8),
				   (3,1,0.1),(3,4,0.1),(4,2,0.25)]

	return createCTMC(transitions,labeling,0,"My_CTMC")


class ModelConverterTestclass(unittest.TestCase):

	def test_MC(var):
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
	
	def test_MDP(var):
		m = modelMDP()
		actions = m.actions
		actions.reverse()
		mstorm = jajapyModeltoStormpy(m)
		mprime = stormpyModeltoJajapy(mstorm,actions)
		mprime.name = m.name
		var.assertEqual(str(mprime),str(m))

	def test_CTMC(var):
		m = modelCTMC()

		mstorm = jajapyModeltoStormpy(m)
		properties = stormpy.parse_properties('T=? [F "blue" ]')
		result = stormpy.check_model_sparse(mstorm,properties[0])
		var.assertAlmostEqual(result.at(mstorm.initial_states[0]),1.0,places=5)

		#mprime = stormpyModeltoJajapy(mstorm)
		#mprime.name = m.name
		#var.assertEqual(str(mprime),str(m))


if __name__ == "__main__":
	unittest.main()