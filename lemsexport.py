from collections import defaultdict
from brian2 import *
from brian2.groups.neurongroup import NeuronGroup
from brian2.equations.equations import PARAMETER, DIFFERENTIAL_EQUATION,\
                                       SUBEXPRESSION
from brian2.core.network import *
import lems.api as lems

from lemsrendering import *
from supporting import read_lems_units, read_lems_dims

import pdb

SPIKE = "spike"

nmlcdpath = ""  # path to NeuroMLCoreDimensions.xml file
lems_dims = read_lems_dims(nmlcdpath=nmlcdpath)
lems_units = read_lems_units(nmlcdpath=nmlcdpath)

def _determine_dimension(value):
    """
    From *value* with Brian2 unit determines proper LEMS dimension.
    """
    for dim in lems_dims:
        if value.has_same_dimensions(lems_dims[dim]):
            return dim
    else:
        raise AttributeError("Dimension not recognized: {}".format(str(value.dim)))

def _to_lems_unit(unit):
    """
    From given unit (and only unit without value!) it return LEMS unit or
    raises exception if unit not supported.
    """
    strunit = unit.in_best_unit()
    strunit = strunit[3:]               # here we substract '1. '
    strunit = strunit.replace('^', '')  # in LEMS there is no ^
    if strunit in lems_units:
        return strunit
    else:
        # in the future we could support adding definition of new unit
        # in that case
        raise AttributeError("Unit not recognized: {}".format(str(strunit)))

def _determine_parameters(paramdict):
    """
    Iterator giving `lems.Parameter` for every parameter from *paramdict*.
    """
    for var in paramdict:
        if is_dimensionless(paramdict[var]):
            yield lems.Parameter(var, "none")
        else:
            dim = _determine_dimension(paramdict[var])
            yield lems.Parameter(var, dim)

def _equation_separator(equation):
    """
    Separates *equation* (str) to LHS and RHS.
    """
    lhs, rhs = equation.split('=')
    return lhs.strip(), rhs.strip()

def create_lems_model(network=None):
    """
    From given *network* returns LEMS model object.
    """
    renderer = LEMSRenderer()
    model = lems.Model()

    if type(network) is not Network:
        net = Network(collect(level=1))
    else:
        net = network
    for obj in net.objects:
        if not type(obj) is NeuronGroup:
            continue
        component = lems.ComponentType(obj.name)
        # adding parameters
        for param in _determine_parameters(obj.namespace):  # in the future we'd like to get rid of namespace
            component.add(param)
        # dynamics of the network
        dynamics = lems.Dynamics()
        ng_equations = obj.equations._equations
        # first step is to extract state and derived variables
        equation_types = defaultdict(list)
        for var in ng_equations:
            equation_types[ng_equations[var].type].append(var)
            if ng_equations[var].type == DIFFERENTIAL_EQUATION:
                sv_ = lems.StateVariable(var,
                                         dimension=_determine_dimension(ng_equations[var].unit)
                                         )
                dynamics.add_state_variable(sv_)
            elif ng_equations[var].type in (PARAMETER, SUBEXPRESSION):
                dv_ = lems.DerivedVariable(var,
                                           dimension=_determine_dimension(ng_equations[var].unit),
                                           value=str(ng_equations[var].expr))
                dynamics.add_derived_variable(dv_)
        # events handling (e.g. spikes)
        for ev in obj.events:
            event_out = lems.EventOut(ev)
            oc = lems.OnCondition(renderer.render_expr(obj.events[ev]))
            oc.add_action(event_out)
            spike_event_eq = _equation_separator(obj.event_codes[ev])
            sa = lems.StateAssignment(spike_event_eq[0], spike_event_eq[1])
            oc.add_action(sa)
            dynamics.add_event_handler(oc)
        # integration regime
        if obj._refractory:
            integr_regime = lems.Regime('integrating', dynamics, True) # True -> initial regime
            dynamics.add_regime(integr_regime)
            # TODO !!!!!!
        for var in equation_types[DIFFERENTIAL_EQUATION]:
            td = lems.TimeDerivative(var, renderer.render_expr(str(ng_equations[var].expr)))
            dynamics.add_time_derivative(td)

        component.dynamics = dynamics
        # adding component to the model
        model.add(component)
    return model

