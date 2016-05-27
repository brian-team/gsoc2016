from lemsrendering import *

def exporter(network=None):
    if (type(network) is not Network):
        net = Network(collect(level=1))
    else:
        net = network
    
    return None
