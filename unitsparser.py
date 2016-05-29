from brian2.units.allunits import all_units
import re

name_to_unit = {u.dispname: u for u in all_units}

def from_string(rep):
    """
    Returns `Quantity` object from text representation of value.

    Parameters
    ----------
    rep : `str`
        text representation of value with unit

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
        raise ValueError("Emtpty value given")
    # match unit
    m = re.match(' ?([a-zA-Z]+)', rep)
    unit = None
    if m:
        unit = rep[0:m.end()].strip()
        rep = rep[m.end():]
    # match exponenet
    m = re.match('-?([0-9]+)?', rep)
    exponent = None
    if len(rep)>0 and m:
        exponent = rep[0:m.end()]
    if value and unit and exponent:
        return float(value) * name_to_unit[unit]**float(exponent)
    elif value and unit:
        return float(value) * name_to_unit[unit]
    else:
        return float(value)

if __name__ == '__main__':
    from brian2 import *
    testlist = ["1 mV", "1.234mV", "1.2e-4 mV", "1.23e-5A", "1.23e4A",
                "1.45E-8 m", "1.23E-8m2", "60", "6000", "123000", "-1.00000008E8"]
    for i in testlist:
        from_string(i)
    print 'ok'
