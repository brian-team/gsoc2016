from brian2 import *
import brian2lems

set_device('neuroml2', filename="ifcgmtest.xml")

n = 100
duration = 1*second
tau = 10*ms

eqs = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''
group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method='linear')
group.v = 0*mV
group.v0 = '20*mV * i / (N-1)'

rec_idx = [2, 63]
statemonitor = StateMonitor(group, 'v', record=rec_idx)
spikemonitor = SpikeMonitor(group, record=rec_idx)

run(duration)
