Brian2NeuroML exporter
======================

This is a short overview of Brian2 supporting package providing functionality of exporting
a model to NeuroML2/LEMS format.

.. contents::
    Overview
    :local:

Working example
---------------
As a demonstration we use simple unconnected Integrate&Fire neurons model with refractoriness
and given initial values.

.. code:: python

    from brian2 import *
    from brian2lems.lemsexport import all_devices

    set_device('neuroml2', filename="nml2model.xml")

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

The use of exporter is pretty straightforward. You need to overwrite ``all_devices`` dictionary
imported from ``brian2`` by importing it from ``brian2lems.lemsexport``.

The next thing is to set a device called ``neuroml2`` which generates NeuroML2/LEMS code.
Note that you need to specify named argument ``filename`` with a name of your model.

.. code:: python

    from brian2lems.lemsexport import all_devices

    set_device('neuroml2', filename="nml2model.xml")

If you use StateMonitor to record some variables, it is transformed to ``Line`` at the ``Display`` of 
NeuroML2 simulation and an ``OutputFile`` tag is added to the model. A name of the output file
is ``recording_<<filename>>.dat``.

SpikeMonitor is parsed to ``EventOutputFile`` with name ``recording_<<filename>>.spikes``.

Limitations
-----------

Currently you should avoid exporting models with:

- synapses

- network input

- multiple runs of simulation

Fixing those issues is in progress.
