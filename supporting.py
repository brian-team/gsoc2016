from brian2.units.allunits import all_units
from brian2 import get_or_create_dimension
from xml.dom.minidom import parse
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


def read_lems_dims(nmlcdpath=""):
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
    domtree = parse(path)
    collection = domtree.documentElement
    dimsCollection = collection.getElementsByTagName("Dimension")
    order_dict = {"m": 1, "l": 0, "t": 2, "i": 3, "k":4, "n":5}  # luminosity ?
    lems_dimensions = dict()
    for dc in dimsCollection:
        name_ = dc.getAttribute("name")
        tmpdim_ = [0]*7  # 7 base dimensions
        for k in order_dict.keys():
            if dc.hasAttribute(k):
                tmpdim_[order_dict[k]] = int(dc.getAttribute(k))
        lems_dimensions[name_] = get_or_create_dimension(tmpdim_)
    return lems_dimensions


def read_lems_units(nmlcdpath=""):
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
    domtree = parse(path)
    collection = domtree.documentElement
    unitsCollection = collection.getElementsByTagName("Unit")
    lems_units = []
    for uc in unitsCollection:
        if uc.hasAttribute('symbol'):
            lems_units.append(uc.getAttribute('symbol'))

    return lems_units

if __name__ == '__main__':
    # test units parser
    from brian2 import *
    testlist = ["1 mV", "1.234mV", "1.2e-4 mV", "1.23e-5A", "1.23e4A",
                "1.45E-8 m", "1.23E-8m2", "60", "6000", "123000",
                "-1.00000008E8", "0.07 per_ms", "10pS"]
    for i in testlist:
        from_string(i)
    print 'ok'
