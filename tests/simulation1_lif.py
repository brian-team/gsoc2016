from brian2 import *
from brian2lems.lemsexport import all_devices
import numpy as np
import argparse

RECORDING_BRIAN_FILENAME = "recording"
LEMS_OUTPUT_FILENAME     = "ifcgmtest.xml"

parser = argparse.ArgumentParser(description='Set simulation flag')
parser.add_argument("-d", "--device", action="store_true",
                    help="runs LEMS output building")
parser.add_argument('--recidx', nargs='+', type=int)

args = parser.parse_args()
flag_device = args.device
if flag_device:
    set_device('lemsdevice')
if args.recidx:
    rec_idx = args.recidx
else:
    rec_idx = [2, 63]

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

statemonitor = StateMonitor(group, 'v', record=rec_idx)
spikemonitor = SpikeMonitor(group, record=rec_idx)

if flag_device:
    run(duration)
    device.build(LEMS_OUTPUT_FILENAME)
else:
    run(duration)
    recvec = []
    for ri in rec_idx:
        recvec.append(statemonitor[ri].v)
    recvec = np.asarray(recvec)
    np.save(RECORDING_BRIAN_FILENAME, recvec)
