from jajapy import *

m_original = loadPrism("/home/raphael/Documents/jajapy/jajapy/tests/materials/pctmc/tl2.pm")
m_original.instantiate(['p','q'],[1.0,4.0])

ts = m_original.generateSet(1000,10,timed=True)

m_initial = loadPrism("/home/raphael/Documents/jajapy/jajapy/tests/materials/pctmc/tl2.pm")

#m_initial.instantiate(['p'],[1.0])
print(m_initial)
m_output = BW_PCTMC().fit(ts,m_initial,stormpy_output=False)
print(m_output)


"""
m_original = loadPrism("/home/raphael/Desktop/sri.pm")
ts = m_original.generateSet(10,10,timed=True)
print(ts.sequences)
m_initial = loadPrism("/home/raphael/Desktop/sri10.pm")
print(m_initial)
m_output = BW_PCTMC().fit_nonInstantiatedParameters(ts,m_initial,stormpy_output=False)
print(m_output)
"""