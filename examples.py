from brian2 import *
from lemsexport import *

# LIF
# http://brian2.readthedocs.io/en/2.0rc1/examples/IF_curve_LIF.html

run_monitor = False
refractory = True

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
                'v0': '20*mV * i / (n-1)'
               }

namespace = {'n':n,
             'tau':tau,
             'init': initializers
            }
if refractory:
    group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                         refractory=15*ms, method='linear', namespace=namespace)
else:
    group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                         method='linear', namespace=namespace)

group.v = 0*mV
group.v0 = '20*mV * i / (n-1)'

model_name = 'lifmodel{}.xml'.format("" if not refractory else "ref")
model = create_lems_model()
model.add(lems.Include("NeuroML2CoreTypes.xml"))
model.add(lems.Include("Simulation.xml"))

model.export_to_file(model_name)


modelstr = open(model_name, 'r').read()
modelstr = modelstr.replace("20*mV * i / (n-1)", "20*mV * 50 / (n-1)")
modelstr = modelstr.replace("</Lems>", """
  <network id="net">
        <population id="neuronpop1" component="n1" size="100"/>
  </network>

  <Simulation id="sim1" length="1s" step="0.1ms" target="net">
  <Display id="d0" title="example trace" timeScale="1ms" xmin="0" xmax="1000" ymin="0" ymax="11">
  <Line id="exampleVoltage" quantity="neuronpop1[50]/v" scale="1mV" timeScale="1ms"/>
  </Display>
  </Simulation>
  <Target component="sim1" />
</Lems>
""")

with open(model_name, 'w') as f:
    f.write(modelstr)

if run_monitor:
    monitor = SpikeMonitor(group)
    run(duration)
    plot(group.v0/mV, monitor.count / duration)
    xlabel('v0 (mV)')
    ylabel('Firing rate (sp/s)')
    show()
