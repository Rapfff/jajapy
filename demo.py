from jajapy import *
from os import remove

def modelPCTMC():
	labeling = ['c_red_p_red','c_green_p_red','c_orange_p_green','c_red_p_green','c_orange_p_red']
	transitions = [(0,1,5.0),(1,2,'p*p/2'),(2,3,2.0),(2,4,0.1),(3,0,0.1),(4,0,2.0)]
	parameters = ['p']
	return createPCTMC(transitions,labeling,parameters,0)

m = modelPCTMC()
m.instantiate(['p'],[2.0])
mprime = jajapyModeltoStormpy(m)
mprime = stormpyModeltoJajapy(mprime)

print(m)
print()
print(mprime)