from jajapy import *
from numpy import array

m_original = loadPrism("/home/raphael/Documents/jajapy/jajapy/tests/materials/pctmc/tl.pm")
m_original.instantiate(['p','q'],[1.0,4.0])

ts = m_original.generateSet(1000,10,timed=True)

m_initial = loadPrism("/home/raphael/Documents/jajapy/jajapy/tests/materials/pctmc/tl.pm")

m_output = BW_PCTMC().fit(ts,m_initial,stormpy_output=False)

print(m_output)
print(m_output.parameter_values)
print(m_original.parameter_values)
print(m_output.transition_expr)
print(m_original.transition_expr)
print(m_original)
