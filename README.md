# gsoc2016
Google Summer of Code project 2016 (Importing and exporting simulator-independent model-descriptions with the Brian simulator)

### Exporter

There are two ways of using Brian2 exporter to LEMS/NML2.

First of all you may use it as a class object, but in that case you need to gather all predefined model attributes into `namespace` and `initializers` dictionaries, as presented below:

```python
from brian2lems.lemsexport import NMLExporter


initializers = {'v': 0*mV,
                'v0': '20*mV * i / (n-1)'
               }

namespace = {'n':n,
             'tau':tau,
            }

################
# Brian2  code #
################

exporter = NMLExporter()
exporter.create_lems_model(initializers=initializers)
model = exporter.model
model.add(lems.Include("NeuroML2CoreTypes.xml"))
model.add(lems.Include("Simulation.xml"))

model.export_to_file("model_name.xml")

```

Second approach is more convenient as it uses **Code Generation Mechanism**. In that case you set a special device and build a LEMS model specifying a name of the output xml file.

```python
from brian2lems.lemsexport import all_devices

set_device('lemsdevice')

################
# Brian2  code #
################

device.build("model_name.xml")
```
