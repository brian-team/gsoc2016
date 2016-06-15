from brian2 import *
from brian2.groups.neurongroup import NeuronGroup
from brian2.equations.equations import PARAMETER, DIFFERENTIAL_EQUATION,\
                                       SUBEXPRESSION
from brian2.core.network import *
import lems.api as lems
import neuroml
import neuroml.writers as writers

from lemsrendering import *
from supporting import read_nml_units, read_nml_dims, brian_unit_to_lems

import pdb

SPIKE             = "spike"
LAST_SPIKE        = "lastspike"
NOT_REFRACTORY    = "not_refractory"
INTEGRATING       = "integrating"
REFRACTORY        = "refractory"
UNLESS_REFRACTORY = "unless refractory"

nmlcdpath = ""  # path to NeuroMLCoreDimensions.xml file
LEMS_CONSTANTS_XML = "LEMSUnitsConstants.xml"  # path to units constants

nml_dims  = read_nml_dims(nmlcdpath=nmlcdpath)
nml_units = read_nml_units(nmlcdpath=nmlcdpath)

renderer = LEMSRenderer()

def _find_precision(value):
    "Returns precision from a float number eg 0.003 -> 0.001"
    splitted = str(value).split('.')
    if len(splitted[0]) > 1:
        return 10**len(splitted[0])
    else:
        return 10**(-1*len(splitted[1]))


def _determine_dimension(value):
    """
    From *value* with Brian2 unit determines proper LEMS dimension.
    """
    for dim in nml_dims:
        if value.has_same_dimensions(nml_dims[dim]):
            return dim
    else:
        raise AttributeError("Dimension not recognized: {}".format(str(value.dim)))


def _to_lems_unit(unit):
    """
    From given unit (and only unit without value!) it returns LEMS unit
    or raises exception if unit not supported.
    """
    strunit = unit.in_best_unit()
    strunit = strunit[3:]               # here we substract '1. '
    strunit = strunit.replace('^', '')  # in LEMS there is no ^
    if strunit in nml_units:
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
        if var == 'init': continue              # initializers (to remove)
        if is_dimensionless(paramdict[var]):
            yield lems.Parameter(var, "none")
        else:
            dim = _determine_dimension(paramdict[var])
            yield lems.Parameter(var, dim)


def _event_builder(events, event_codes):
    """
    From *events* and *event_codes* yields lems.OnCondition objects
    """
    for ev in events:
        event_out = lems.EventOut(ev)  # output (e.g. spike)
        oc = lems.OnCondition(renderer.render_expr(events[ev]))
        oc.add_action(event_out)
        event_eq = _equation_separator(event_codes[ev])
        sa = lems.StateAssignment(event_eq[0], event_eq[1])
        oc.add_action(sa)
        spike_flag = False
        if ev == SPIKE:
            spike_flag = True
        yield (spike_flag, oc)


def _equation_separator(equation):
    """
    Separates *equation* (str) to LHS and RHS.
    """
    lhs, rhs = equation.split('=')
    return lhs.strip(), rhs.strip()


def create_lems_model(network=None, constants_file=None):
    """
    From given *network* returns LEMS model object.
    """
    model = lems.Model()

    if type(network) is not Network:
        net = Network(collect(level=1))
    else:
        net = network

    if not constants_file:
        model.add(lems.Include(LEMS_CONSTANTS_XML))
    else:
        model.add(lems.Include(constants_file))

    for e, obj in enumerate(net.objects):
        if not type(obj) is NeuronGroup:
            continue
        ct_name = "neuron{}".format(e+1)
        component_type = lems.ComponentType(ct_name)
        # adding parameters
        for param in _determine_parameters(obj.namespace):  # in the future we'd like to get rid of namespace
            component_type.add(param)
        # common things for every neuron definition
        component_type.add(lems.EventPort(name='spike', direction='out'))
        # dynamics of the network
        dynamics = lems.Dynamics()
        ng_equations = obj.user_equations
        for var in ng_equations:
            if ng_equations[var].type == DIFFERENTIAL_EQUATION:
                dim_ = _determine_dimension(ng_equations[var].unit)
                sv_ = lems.StateVariable(var, dimension=dim_, exposure=var)
                dynamics.add_state_variable(sv_)
                component_type.add(lems.Exposure(var, dimension=dim_))
            elif ng_equations[var].type in (PARAMETER, SUBEXPRESSION):
                if var == NOT_REFRACTORY:
                    continue
                dim_ = _determine_dimension(ng_equations[var].unit)
                sv_ = lems.StateVariable(var, dimension=dim_)
                dynamics.add_state_variable(sv_)
        # what happens at initialization
        onstart = lems.OnStart()
        for var in obj.equations.names:
            if var in (NOT_REFRACTORY, LAST_SPIKE):
                continue
            init_value = obj.namespace['init'][var]  # initializers will be removed in the future
            if type(init_value) != str:
                init_value = brian_unit_to_lems(init_value)
            onstart.add(lems.StateAssignment(var, init_value))
        dynamics.add(onstart)

        if obj._refractory:
            # if refractoriness, we create separate regimes
            # - for integrating
            integr_regime = lems.Regime(INTEGRATING, dynamics, True)  # True -> initial regime
            for spike_flag, oc in _event_builder(obj.events, obj.event_codes):
                if spike_flag:
                    # if spike occured we make transition to refractory regime
                    oc.add_action(lems.Transition(REFRACTORY))
                integr_regime.add_event_handler(oc)
            # - for refractory
            refrac_regime = lems.Regime(REFRACTORY, dynamics)
            # we make lastspike variable and initialize it
            refrac_regime.add_state_variable(lems.StateVariable(LAST_SPIKE, dimension='time'))
            oe = lems.OnEntry()
            oe.add(lems.StateAssignment(LAST_SPIKE, 't'))
            refrac_regime.add(oe)
            # after time spiecified in _refractory we make transition
            # to integrating regime
            ref_oc = lems.OnCondition('t .gt. ( {0} + {1} )'.format(LAST_SPIKE, brian_unit_to_lems(obj._refractory)))
            ref_trans = lems.Transition(INTEGRATING)
            ref_oc.add_action(ref_trans)
            refrac_regime.add_event_handler(ref_oc)
            for var in obj.user_equations.diff_eq_names:
                td = lems.TimeDerivative(var, renderer.render_expr(str(ng_equations[var].expr)))
                # if unless refratory we add only do integration regime
                if UNLESS_REFRACTORY in ng_equations[var].flags:
                    integr_regime.add_time_derivative(td)
                    continue
                integr_regime.add_time_derivative(td)
                refrac_regime.add_time_derivative(td)
            dynamics.add_regime(integr_regime)
            dynamics.add_regime(refrac_regime)
        else:
            # here we add events directly to dynamics
            for spike_flag, oc in _event_builder(obj.events, obj.event_codes):
                dynamics.add_event_handler(oc)
            for var in obj.user_equations.diff_eq_names:
                td = lems.TimeDerivative(var, renderer.render_expr(str(ng_equations[var].expr)))
                dynamics.add_time_derivative(td)

        component_type.dynamics = dynamics
        # adding component to the model
        model.add_component_type(component_type)
        obj.namespace.pop("init", None)                # filter out init
        model.add(lems.Component("n{}".format(e+1), ct_name, **obj.namespace))
    return model


def create_nml_network(include, nml_file='name.xml'):
    nml_doc = neuroml.NeuroMLDocument()
    nml_doc.includes.append(neuroml.Include(include))
    network = neuroml.Network(id='net')
    pop = neuroml.Population(id='neuropop', component='n1', size='100')
    network.populations.append(pop)
    nml_doc.networks.append(network)
    writers.NeuroMLWriter.write(nml_doc, nml_file)
    return None
