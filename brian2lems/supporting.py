from brian2.units.allunits import all_units
from brian2 import get_or_create_dimension
import xml.dom.minidom as minidom

import re

name_to_unit = {u.dispname: u for u in all_units}


def from_string(rep):
    """
    Returns `Quantity` object from text representation of a value.

    Parameters
    ----------
    rep : `str`
        text representation of a value with unit

    Returns
    -------
    q : `Quantity`
        Brian Quantity object
    """
    # match value
    m = re.match('-?[0-9]+\.?([0-9]+)?[eE]?-?([0-9]+)?', rep)
    if m:
        value = rep[0:m.end()]
        rep = rep[m.end():]
    else:
        raise ValueError("Empty value given")
    # match unit
    m = re.match(' ?([a-zA-Z]+)', rep)
    unit = None
    per = None
    if m:
        unit = rep[0:m.end()].strip()
        # special case with per
        if unit=='per':
            mper = re.match(' ?per_([a-zA-Z]+)', rep)
            per = rep[0:mper.end()].strip()[4:]
            m = mper
        rep = rep[m.end():]
    # match exponent
    m = re.match('-?([0-9]+)?', rep)
    exponent = None
    if len(rep) > 0 and m:
        exponent = rep[0:m.end()]
    if unit:
        if per:
            b2unit = 1. / name_to_unit[per]
        else:
            b2unit = name_to_unit[unit]
        if value and exponent:
            return float(value) * b2unit**float(exponent)
        elif value:
            return float(value) * b2unit
    else:
        return float(value)

def brian_unit_to_lems(valunit):
    'Simply adds * between value and unit, e.g. "20. mV" -> "20.*mV"'
    if float(valunit)==0:
        return '0'
    if type(valunit)!=str:
        valunit = str(valunit)
    return valunit.replace(' ', '*')

def read_nml_dims(nmlcdpath=""):
    """
    Read from `NeuroMLCoreDimensions.xml` all supported by LEMS
    dimensions and store it as a Python dict with name as a key
    and Brian2 unit as value.

    Parameters
    ----------
    nmlcdpath : `str`, optional
        Path to 'NeuroMLCoreDimensions.xml'

    Returns
    -------
    lems_dimenssions : `dict`
        Dictionary with LEMS dimensions.
    """
    path = nmlcdpath + "NeuroMLCoreDimensions.xml"
    domtree = minidom.parse(path)
    collection = domtree.documentElement
    dimsCollection = collection.getElementsByTagName("Dimension")
    order_dict = {"m": 1, "l": 0, "t": 2, "i": 3, "k":4, "n":5, "j":6}
    lems_dimensions = dict()
    for dc in dimsCollection:
        name_ = dc.getAttribute("name")
        tmpdim_ = [0]*7  # 7 base dimensions
        for k in order_dict.keys():
            if dc.hasAttribute(k):
                tmpdim_[order_dict[k]] = int(dc.getAttribute(k))
        lems_dimensions[name_] = get_or_create_dimension(tmpdim_)
    return lems_dimensions


def read_nml_units(nmlcdpath=""):
    """
    Read from 'NeuroMLCoreDimensions.xml' all supported by LEMS
    units.

    Parameters
    ----------
    nmlcdpath : `str`, optional
        Path to 'NeuroMLCoreDimensions.xml'

    Returns
    -------
    lems_units : `list`
        List with LEMS units.
    """
    path = nmlcdpath + "NeuroMLCoreDimensions.xml"
    domtree = minidom.parse(path)
    collection = domtree.documentElement
    unitsCollection = collection.getElementsByTagName("Unit")
    lems_units = []
    for uc in unitsCollection:
        if uc.hasAttribute('symbol'):
            lems_units.append(uc.getAttribute('symbol'))
    return lems_units

class NeuroMLSimulation(object):
    
    def __init__(self, simid, target, length="1s", step="0.1ms"):
        self.doc = minidom.Document()
        self.create_simulation(simid, target, length, step)
        self.lines = []

    def create_simulation(self, simid, target, length, step):
        self.simulation = self.doc.createElement('Simulation')
        attributes = [("id", simid), ("target", target),
                      ("length", length), ("step", step)]
        for attr_name, attr_value in attributes:
            self.simulation.setAttribute(attr_name, attr_value)
        

    def add_display(self, dispid, title="", time_scale="1ms", xmin="0",
                                  xmax="1000", ymin="0", ymax="11"):
        self.display = self.doc.createElement('Display')
        attributes = [("id", dispid), ("title", title),
                      ("timeScale", time_scale), ("xmin", xmin),
                      ("xmax", xmax), ("ymin", ymin), ("ymax", ymax)]
        for attr_name, attr_value in attributes:
            self.display.setAttribute(attr_name, attr_value)

    def add_line(self, linid, quantity, scale="1mV", time_scale="1ms"):
        assert hasattr(self, 'display'), "You need to add display first"
        line = self.doc.createElement('Line')
        attributes = [("id", linid), ("quantity", quantity),
                      ("scale", scale), ("timeScale", time_scale)]
        for attr_name, attr_value in attributes:
            line.setAttribute(attr_name, attr_value)
        self.lines.append(line)

    def build(self):
        """
        Build NeuroML DOM structure of Simulation.
        """
        for line in self.lines:
            self.display.appendChild(line)
        self.simulation.appendChild(self.display)
        return self.simulation

    def __repr__(self):
        return self.simulation.toprettyxml('  ', '\n')

if __name__ == '__main__':
    # test units parser
    from brian2 import *
    testlist = ["1 mV", "1.234mV", "1.2e-4 mV", "1.23e-5A", "1.23e4A",
                "1.45E-8 m", "1.23E-8m2", "60", "6000", "123000",
                "-1.00000008E8", "0.07 per_ms", "10pS"]
    for i in testlist:
        from_string(i)
    print 'ok'

    # test NeuroMLSimulation
    nmlsim = NeuroMLSimulation('a', 'b')
    nmlsim.add_display('ex')
    nmlsim.add_line('line1', 'v')
    nmlsim.build()
