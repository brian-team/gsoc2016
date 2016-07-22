from brian2 import *
import lems.api as lems
from brian2lems.lemsexport import *
from brian2lems.lemsexport import NMLExporter

# LIF
# http://brian2.readthedocs.io/en/2.0rc1/examples/IF_curve_LIF.html

show = False
refractory = False

n = 100
duration = 5*second
tau = 10*ms

if refractory:
    eqs = '''
    dv/dt = (v0 - v) / tau : volt (unless refractory)
    v0 : volt
    '''
else:
    eqs = '''
    dv/dt = (v0 - v) / tau : volt
    v0 : volt
    '''

initializers = {'v': 0*mV,
                'v0': '20*mV * i / (N-1)'
               }

namespace = {'n':n,
             'tau':tau,
            }
if refractory:
    group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                         refractory=15*ms, method='linear', namespace=namespace)
else:
    group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                         method='linear', namespace=namespace)

group.v = 0*mV
group.v0 = '20*mV * i / (N-1)'


statemonitor = StateMonitor(group, 'v', record=[2,63])
spikemonitor = SpikeMonitor(group, record=[2,63])

model_name = 'lifmodel{}.xml'.format("" if not refractory else "ref")

exporter = NMLExporter()
exporter.create_lems_model(initializers=initializers, includes=["NeuroML2CoreTypes.xml","Simulation.xml"])
exporter.export_to_file(model_name)

if show:
    plot(group.v0/mV, monitor.count / duration)
    xlabel('v0 (mV)')
    ylabel('Firing rate (sp/s)')
    show()
