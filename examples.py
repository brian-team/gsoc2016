from brian2 import *
import re
eqs = '''
      dv/dt = (v0 - v) / tau : volt
      v0 : volt
      '''
n = 100
params = { 'duration': 1*second,
           'tau': 10*ms}

group = NeuronGroup(n, eqs,
                    threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, namespace=params)
group.v = 0*mV
group.v0 = '20*mV * i / (n-1)'

