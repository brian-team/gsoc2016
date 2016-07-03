from brian2 import *
from brian2.groups.neurongroup import NeuronGroup
from brian2.equations.equations import PARAMETER, DIFFERENTIAL_EQUATION,\
                                       SUBEXPRESSION
from brian2.core.network import *
from brian2.core.namespace import get_local_namespace, DEFAULT_UNITS
from brian2.devices.device import Device, all_devices
from brian2.utils.logger import get_logger
from brian2.units.fundamentalunits import _siprefixes

import lems.api as lems
import neuroml
import neuroml.writers as writers

from lemsrendering import *
from supporting import read_nml_units, read_nml_dims, brian_unit_to_lems,\
                       name_to_unit
from cgmhelper import *

import re
import pdb

__all__ = []

logger = get_logger(__name__)

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
        if value==1:
            # dimensionless
            return "none"
        else:
            raise AttributeError("Dimension not recognized: {}".format(str(value.dim)))


def _to_lems_unit(unit):
    """
    From given unit (and only unit without value!) it returns LEMS unit
    """
    if type(unit) == str:
        strunit = unit
    else:
        strunit = unit.in_best_unit()
        strunit = strunit[3:]               # here we substract '1. '
    strunit = strunit.replace('^', '')  # in LEMS there is no ^
    return strunit


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
    try:
        lhs, rhs = re.split('<=|>=|==|=|>|<', equation)
    except ValueError:
        return None
    return lhs.strip(), rhs.strip()


def make_lems_unit(newunit):
    """
    Returns from *newunit* to a lems.Unit definition.
    """
    strunit = _to_lems_unit(newunit)
    power = int(np.log10((mmetre**2).base))
    dimension = _determine_dimension(newunit)
    return lems.Unit(strunit, symbol=strunit, dimension=dimension, power=power)


class NMLExporter(object):
    """
    Exporter from Brian2 code to NeuroML.
    """
    def __init__(self):
        self._model = lems.Model()

    def _unit_lems_validator(self, value_in_unit):
        """
        Checks if *unit* is in NML supported units and if it is not
        it adds a new definition to model. Eventually returns value
        with unit in string.
        """
        if is_dimensionless(value_in_unit):
            return str(value_in_unit)
        value, unit = value_in_unit.in_best_unit().split(' ')
        lemsunit = _to_lems_unit(unit)
        if lemsunit in nml_units:
            return "{} {}".format(value, lemsunit)
        else:
            self._model.add(make_lems_unit(name_to_unit[unit]))
            return "{} {}".format(value, lemsunit)

    def _event_builder(self, events, event_codes):
        """
        From *events* and *event_codes* yields lems.OnCondition objects
        """
        for ev in events:
            event_out = lems.EventOut(ev)  # output (e.g. spike)
            oc = lems.OnCondition(renderer.render_expr(events[ev]))
            oc.add_action(event_out)
            # if event is not in model ports we should add it
            if not ev in self._component_type.event_ports:
                self._component_type.add(lems.EventPort(name=ev, direction='out'))
            if ev in event_codes:
                for ec in re.split(';|\n', event_codes[ev]):
                    event_eq = _equation_separator(ec)
                    oc.add_action(lems.StateAssignment(event_eq[0], event_eq[1]))
            spike_flag = False
            if ev == SPIKE:
                spike_flag = True
            yield (spike_flag, oc)

    def create_lems_model(self, network=None, initializers={}, constants_file=None):
        """
        From given *network* returns LEMS model object.
        """
        if network is None:
            net = Network(collect(level=1))
        else:
            net = network

        if not constants_file:
            self._model.add(lems.Include(LEMS_CONSTANTS_XML))
        else:
            self._model.add(lems.Include(constants_file))

        for e, obj in enumerate(net.objects):
            if not type(obj) is NeuronGroup:
                continue
            ct_name = "neuron{}".format(e+1)
            self._component_type = lems.ComponentType(ct_name)
            # adding parameters
            for param in _determine_parameters(obj.namespace):  # in the future we'd like to get rid of namespace
                self._component_type.add(param)
            # common things for every neuron definition
            self._component_type.add(lems.EventPort(name='spike', direction='out'))
            # dynamics of the network
            dynamics = lems.Dynamics()
            ng_equations = obj.user_equations
            for var in ng_equations:
                if ng_equations[var].type == DIFFERENTIAL_EQUATION:
                    dim_ = _determine_dimension(ng_equations[var].unit)
                    sv_ = lems.StateVariable(var, dimension=dim_, exposure=var)
                    dynamics.add_state_variable(sv_)
                    self._component_type.add(lems.Exposure(var, dimension=dim_))
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
                if not var in initializers:
                    continue
                init_value = initializers[var]  # initializers will be removed in the future
                if type(init_value) != str:
                    init_value = brian_unit_to_lems(init_value)
                onstart.add(lems.StateAssignment(var, init_value))
            dynamics.add(onstart)

            if obj._refractory:
                # if refractoriness, we create separate regimes
                # - for integrating
                integr_regime = lems.Regime(INTEGRATING, dynamics, True)  # True -> initial regime
                for spike_flag, oc in self._event_builder(obj.events, obj.event_codes):
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
                if not _equation_separator(str(obj._refractory)):
                    # if there is no specific variable given, we assume
                    # that this is time condition
                    ref_oc = lems.OnCondition('t .gt. ( {0} + {1} )'.format(LAST_SPIKE, brian_unit_to_lems(obj._refractory)))
                else:
                    ref_oc = lems.OnCondition(renderer.render_expr(obj._refractory))
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

            self._component_type.dynamics = dynamics
            # making componenttype is done so we add it to the model
            self._model.add_component_type(self._component_type)
            obj.namespace.pop("init", None)                # kick out init
            # adding component to the model
            paramdict = dict()
            for param in obj.namespace:
                paramdict[param] = self._unit_lems_validator(obj.namespace[param])
            self._model.add(lems.Component("n{}".format(e+1), ct_name, **paramdict))

    @property
    def model(self):
        return self._model


class DummyCodeObject(object):
    def __init__(self, *args, **kwds):
        pass

    def __call__(self, **kwds):
        pass


class LEMSDevice(Device):
    '''
    The `Device` used LEMS/NeuroML2 code genration.
    '''
    def __init__(self):
        super(LEMSDevice, self).__init__()
        self.runs = []
        self.assignments = []

    # Our device will not actually calculate/store any data, so the following
    # are just dummy implementations

    def add_array(self, var):
        pass

    def init_with_zeros(self, var, dtype):
        pass

    def fill_with_array(self, var, arr):
        pass

    def init_with_arange(self, var, start, dtype):
        pass

    def get_value(self, var, access_data=True):
        return np.zeros(var.size, dtype=var.dtype)

    def resize(self, var, new_size):
        pass

    def code_object(self, *args, **kwds):
        return DummyCodeObject(*args, **kwds)

    def network_run(self, network, duration, report=None, report_period=10*second,
                    namespace=None, profile=True, level=0):
        network._clocks = {obj.clock for obj in network.objects}
        # Get the local namespace
        if namespace is None:
            namespace = get_local_namespace(level=level+2)
        network.before_run(namespace)

        # Extract all the objects present in the network
        descriptions = []
        merged_namespace = {}
        self.network = network  
        for obj in network.objects:
            one_description, one_namespace = description(obj, namespace)
            descriptions.append((obj.name, one_description))
            for key, value in one_namespace.iteritems():
                if key in merged_namespace and value != merged_namespace[key]:
                    raise ValueError('name "%s" is used inconsistently')
                merged_namespace[key] = value

        assignments = list(self.assignments)
        self.assignments[:] = []
        self.runs.append((descriptions, duration, merged_namespace, assignments))

    def variableview_set_with_expression_conditional(self, variableview, cond, code,
                                                     run_namespace, check_units=True):
        self.assignments.append(('conditional', variableview.group.name, variableview.name, cond, code))

    def variableview_set_with_expression(self, variableview, item, code, run_namespace, check_units=True):
        self.assignments.append(('item', variableview.group.name, variableview.name, item, code))

    def variableview_set_with_index_array(self, variableview, item, value, check_units):
        self.assignments.append(('item', variableview.group.name, variableview.name, item, value))

    def build(self):
        print self.network.objects 
        print len(self.runs[1])
        initializers = {}
        for descriptions, duration, namespace, assignments in self.runs:
            print '*'*8
            print descriptions
            print duration
            print namespace
            print assignments
            for a in assignments:
                if not a[2] in initializers:
                    initializers[a[2]] = a[-1]
        print 'x '*8
        print initializers
        print type(self.network.objects[0])
        exporter = NMLExporter()
        exporter.create_lems_model(self.network)
        model = exporter.model
        model.export_to_file("xxx.xml")


lems_device = LEMSDevice()
all_devices['lemsdevice'] = lems_device
