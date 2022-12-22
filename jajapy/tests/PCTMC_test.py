import unittest
from ..pctmc import *
from os import remove
from ..base.Set import *
from math import log
from numpy import where, array

def modelPCTMC_ped():
	labeling = ['red','green']
	transitions = [(1,0,5.0)]
	sync_trans = [(0,'b',1,0.5),(0,'r',0,10.0),(0,'g',0,10.0)]
	parameters = ['r','g']
	return createPCTMC(transitions,labeling,parameters,0,{},sync_trans)

def modelPCTMC_car2():
	labeling = ['red','green']
	transitions = []
	sync_trans = [(0,'g',1,'g'),(1,'r',0,'r'),(0,'b',0,10.0),(1,'b',0,2.0)]
	parameters = ['r','g']
	return createPCTMC(transitions,labeling,parameters,1,{},sync_trans)

def modelPCTMC_car1():
	labeling = ['red','green']
	transitions = []
	sync_trans = [(0,'r',1,'r'),(1,'g',0,'g'),(0,'b',0,10.0),(1,'b',0,2.0)]
	parameters = ['r','g']
	return createPCTMC(transitions,labeling,parameters,0,{},sync_trans)

class PCTMCTestclass(unittest.TestCase):

	def test_PCTMC_fast(var):
		m1 = modelPCTMC_car1()
		m1.instantiate(['r'],[0.1])
		var.assertFalse(m1.isInstantiated())
		m1.instantiate(['g'],[0.5])
		var.assertTrue(m1.isInstantiated())
		m1 = modelPCTMC_car1()
		m1.instantiate(['r','g'],[0.1,0.5])
		var.assertTrue(m1.isInstantiated())
		
		


if __name__ == "__main__":
	unittest.main()