# gsoc2016

Google Summer of Code project 2016 (Importing and exporting simulator-independent model-descriptions with the Brian simulator)

### brian2lems

The use of exporter to LEMS/NeuroML2 is very straightforward. Just set new device `neuroml2` and specify the name of the output XML file.

```python

from brian2 import *
import brian2lems

set_device('neuroml2', filename="nml2model.xml")

################
# Brian2  code #
################

```

More details you may find in user or developer documentation.
