from brian2 import *
from lemsexport import *

# LIF
# http://brian2.readthedocs.io/en/2.0rc1/examples/IF_curve_LIF.html

n = 100
duration = 1*second
tau = 10*ms

eqs = '''
dv/dt = (v0 - v) / tau : volt 
v0 : volt
'''

namespace = {'n':n,
             'duration':duration,
             'tau':tau,
             }

group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    method='linear', namespace=namespace)

group.v = 0*mV
group.v0 = '20*mV * i / (n-1)'

#model = create_lems_model()
#model.include_file("NeuroMLCoreDimensions.xml", ["NeuroML2CoreTypes"])
#model.export_to_file('abc.xml')

#monitor = SpikeMonitor(group)

#run(duration)
#plot(group.v0/mV, monitor.count / duration)
#xlabel('v0 (mV)')
#ylabel('Firing rate (sp/s)')
#show()
