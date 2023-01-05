from jajapy import *

m_original = loadPrism("/home/raphael/Documents/jajapy/jajapy/tests/materials/pctmc/tl2.pm")
m_original.instantiate(['p','q'],[1.0,4.0])

ts = m_original.generateSet(1000,10,timed=True)

m_initial = loadPrism("/home/raphael/Documents/jajapy/jajapy/tests/materials/pctmc/tl2.pm")

m_output = BW_PCTMC().fit(ts,m_initial,stormpy_output=False)

print(m_output)
