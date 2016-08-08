from brian2 import *
from brian2.groups.neurongroup import NeuronGroup, Thresholder, Resetter,\
                                      StateUpdater
from brian2.equations.equations import PARAMETER, DIFFERENTIAL_EQUATION,\
                                       SUBEXPRESSION
from brian2.core.network import *
from brian2.core.namespace import get_local_namespace, DEFAULT_UNITS
from brian2.devices.device import Device, RuntimeDevice, all_devices
from brian2.utils.logger import get_logger
from brian2.units.fundamentalunits import _siprefixes
from brian2.utils.stringtools import get_identifiers

import lems.api as lems
import neuroml
import neuroml.writers as writers

from lemsrendering import *
from supporting import read_nml_units, read_nml_dims, brian_unit_to_lems,\
                       name_to_unit, NeuroMLSimulation, NeuroMLSimpleNetwork,\
                       NeuroMLTarget
from cgmhelper import *

import numpy as np
import warnings
import re
import os
import pdb

__all__ = []

logger = get_logger(__name__)

SPIKE             = "spike"
LAST_SPIKE        = "lastspike"
NOT_REFRACTORY    = "not_refractory"
INTEGRATING       = "integrating"
REFRACTORY        = "refractory"
UNLESS_REFRACTORY = "unless refractory"
INDEX             = "index"    # iterator in LEMS
BASE_CELL         = "baseCell"
BASE_POPULATION   = "basePopulation"

nmlcdpath = os.path.dirname(__file__)  # path to NeuroMLCoreDimensions.xml file
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
        self._all_params_unit = {}
        self._population = None
        self._model_namespace = {'neuronname': None,
                                 'ct_populationname': None,
                                 'populationname': None,
                                 'networkname' : None,
                                 'targetname': None,
                                 'simulname': None}

    def _determine_parameters(self, paramdict):
        """
        Iterator giving `lems.Parameter` for every parameter from *paramdict*.
        """
        for var in paramdict:
            if is_dimensionless(paramdict[var]):
                self._all_params_unit[var] = "none"
                yield lems.Parameter(var, "none")
            else:
                dim = _determine_dimension(paramdict[var])
                self._all_params_unit[var] = dim
                yield lems.Parameter(var, dim)

    def _determine_properties(self, paramdict):
        """
        Iterator giving `lems.Property` for every parameter from *paramdict*.
        """
        for var in paramdict:
            if is_dimensionless(paramdict[var]):
                self._all_params_unit[var] = "none"
                yield lems.Property(var, "none")
            else:
                dim = _determine_dimension(paramdict[var])
                self._all_params_unit[var] = dim
                yield lems.Property(var, dim)

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
    
    def add_neurongroup(self, obj, idx_of_ng, namespace, initializers):
        """
        Adds NeuronGroup object *obj* to self._model.
        *obj* -- NeuronGroup object
        """
        if hasattr(obj, "namespace") and not obj.namespace:
            obj.namespace = namespace
        self._nr_of_neurons = obj.N # maybe not the most robust solution
        ct_name = "neuron{}".format(idx_of_ng+1)
        self._model_namespace["neuronname"] = ct_name
        self._component_type = lems.ComponentType(ct_name, extends=BASE_CELL)
        # adding parameters
        special_properties = {}
        for key in obj.namespace.keys():
            special_properties[key] = None
        for param in self._determine_properties(obj.namespace):
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
                self._all_params_unit[var] = dim_
                dynamics.add_state_variable(sv_)
                self._component_type.add(lems.Exposure(var, dimension=dim_))
            elif ng_equations[var].type in (PARAMETER, SUBEXPRESSION):
                if var == NOT_REFRACTORY:
                    continue
                dim_ = _determine_dimension(ng_equations[var].unit)
                self._all_params_unit[var] = dim_
                # all initializers contatining iterator need to be assigned
                # as a property
                # i is default iterator in Brian2
                if var in initializers and "i" in get_identifiers(str(initializers[var])):
                    self._component_type.add(lems.Property(var, dim_))
                    special_properties[var] = initializers[var]
                    continue
                sv_ = lems.StateVariable(var, dimension=dim_)
                dynamics.add_state_variable(sv_)
        # what happens at initialization
        onstart = lems.OnStart()
        for var in obj.equations.names:
            if var in (NOT_REFRACTORY, LAST_SPIKE):
                continue
            if not var in initializers:
                continue
            if var in special_properties:
                continue
            init_value = initializers[var]
            if type(init_value) != str:
                init_value = brian_unit_to_lems(init_value)
            else:
                init_value = renderer.render_expr(str(init_value))
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
            for spike_flag, oc in self._event_builder(obj.events, obj.event_codes):
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
        if obj.N == 1:
            self._model.add(lems.Component("n{}".format(idx_of_ng), ct_name, **paramdict))
        else:
            self.make_multiinstantiate(special_properties, ct_name, paramdict)

    def make_multiinstantiate(self, special_properties, name, parameters):
        """
        Adds ComponentType with MultiInstantiate in order to make
        a population of neurons.
        """
        PARAM_SUBSCRIPT = "_p"
        self._model_namespace["ct_populationname"] = name+"Multi"
        multi_ct = lems.ComponentType(self._model_namespace["ct_populationname"], extends=BASE_POPULATION)
        structure = lems.Structure()
        multi_ins = lems.MultiInstantiate(component_type=name,
                                          number="N")
        param_dict = {}
        # number of neruons
        multi_ct.add(lems.Parameter(name="N", dimension="none"))
        # other parameters
        for sp in special_properties:
            if special_properties[sp] is None:
                multi_ct.add(lems.Parameter(name=sp+PARAM_SUBSCRIPT, dimension=self._all_params_unit[sp]))
                multi_ins.add(lems.Assign(property=sp, value=sp+PARAM_SUBSCRIPT))
                param_dict[sp] = parameters[sp]
            else:
                # multi_ct.add(lems.Parameter(name=sp, dimension=self._all_params_unit[sp]))
                # check if there are some units in equations
                equation = special_properties[sp]
                for i in get_identifiers(equation):
                    # iterator is a special case
                    if i == "i":
                        regexp_noletter = "[^a-zA-Z0-9]"
                        equation = re.sub("{re}i{re}".format(re=regexp_noletter),\
                                                  " {} ".format(INDEX), equation)
                    # here it's assumed that we don't use Netwton in neuron models
                    elif i in name_to_unit and i != "N":
                        const_i = i+'const'
                        multi_ct.add(lems.Constant(name=const_i, symbol=const_i,
                                     dimension=self._all_params_unit[sp], value="1"+i))
                        equation = re.sub(i, const_i, equation)
                multi_ins.add(lems.Assign(property=sp, value=equation))
        structure.add(multi_ins)
        multi_ct.structure = structure
        self._model.add(multi_ct)
        param_dict = dict([(k+"_p", v) for k, v in param_dict.items()])
        param_dict["N"] = self._nr_of_neurons
        #self._model.add(lems.Component(name+"Multi", name+"network", **param_dict))
        self._model_namespace["populationname"] = self._model_namespace["ct_populationname"] + "pop"
        self._model_namespace["networkname"] = self._model_namespace["ct_populationname"] + "Net"
        self.add_population(self._model_namespace["networkname"],
                            self._model_namespace["populationname"],
                            self._model_namespace["ct_populationname"],
                            **param_dict)

    def add_statemonitor(self, obj, filename="recording", outputfile=False):
        """
        From StateMonitor object extracts indices to recording in LEMS 
        simulation and makes a display.
        *obj* -- StateMonitor object
        *filename*  -- name of output file without extension
        *outputfile* -- flag sayinf whether to record output to file
        """
        filename += '.dat'
        indices = np.asarray(obj.record).copy()
        if isinstance(indices, bool) and indices == True:
            indices = np.arange(self._nr_of_neurons)
        indices += 1
        variables = obj.needed_variables
        # step of integration
        dt = str(obj.clock.dt.in_unit(ms))
        self._simulation.update_simulation_attribute('step', dt)
        # adding display and outputcolumn for each recorded neuron
        for e, var in enumerate(variables):
            self._simulation.add_display("disp{}".format(e), str(var)) # max, min etc ???
            if outputfile:
                self._simulation.add_outputfile("of{}".format(e), filename=filename)
            for i in indices:
                # scale, time_scale ???
                self._simulation.add_line("line{}".format(i),
                                          "{}[{}]/v".format(self._model_namespace["populationname"],i))
                if outputfile:
                    self._simulation.add_outputcolumn("{}".format(i),
                                                      "{}[{}]/v".format(self._model_namespace["populationname"],i))

    def add_spikemonitor(self, obj, filename="recording"):
        """
        From SpikeMonitor object extracts indices to recording in LEMS 
        simulation and makes a display.
        *obj* -- SpikeMonitor object
        *filename*  -- name of output file without extension
        """
        filename += '.spikes'
        indices = np.asarray(obj.record).copy()
        if isinstance(indices, bool) and indices == True:
            indices = np.arange(self._nr_of_neurons)
        indices += 1
        variables = obj.needed_variables
        self._simulation.add_eventoutputfile("eof", filename)
        # adding eventselection for each recorded neuron
        for i in indices:
            self._simulation.add_eventselection("line{}".format(i),
                    "{}[{}]".format(self._model_namespace["populationname"],i),
                    event_port = "spike")

    def add_synapses(self, obj):
        """
        Adds synapses to the model.
        """
        pass

    def add_population(self, net_id, component_id, type_, **args):
        """
        Sets population of neurons to resulting file.
        """
        nmlnetwork = NeuroMLSimpleNetwork(net_id)
        nmlnetwork.add_component(component_id, type_, **args)
        self._population = nmlnetwork.build()

    def add_include(self, includefile):
        """
        Adds file to include *includefile* to model.
        *includefile* -- str
        """
        self._model.add(lems.Include(includefile))
    
    def create_lems_model(self, network=None, namespace={}, initializers={},
                                           constants_file=None, includes=[],
                                           recordingsname='recording'):
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
        includes = set(includes)
        for incl in INCLUDES:
            includes.add(incl)
        for incl in includes:
            self.add_include(incl)
        neuron_groups  = [o for o in net.objects if type(o) is NeuronGroup]
        state_monitors = [o for o in net.objects if type(o) is StateMonitor]
        spike_monitors = [o for o in net.objects if type(o) is SpikeMonitor]
        synapses       = [o for o in net.objects if type(o) is Synapses]

        # Thresholder, Resetter, StateUpdater are not interesting from our perspective

        # First step is to add individual neuron deifinitions and initialize
        # them by MultiInstantiate
        for e, obj in enumerate(neuron_groups):
            self.add_neurongroup(obj, e, namespace, initializers)
        # DOM structure of the whole model is constructed below
        self._dommodel = self._model.export_to_dom()
        # A population should be created in *make_multiinstantiate*
        # so we can add it to our DOM structure.
        if self._population:
            self._extend_dommodel(self._population)
        # if some State or Spike Monitors occur we support them by
        # Simulation tag
        self._model_namespace['simulname'] = "sim1"
        self._simulation = NeuroMLSimulation(self._model_namespace['simulname'],
                                             self._model_namespace['networkname'])
        
        for e, obj in enumerate(state_monitors):
            self.add_statemonitor(obj, filename=recordingsname, outputfile=True)
        for e, obj in enumerate(spike_monitors):
            self.add_spikemonitor(obj, filename=recordingsname)
        simulation = self._simulation.build()
        self._extend_dommodel(simulation)
        target = NeuroMLTarget(self._model_namespace['simulname'])
        target = target.build()
        self._extend_dommodel(target)
        #message = "{} currently not supported ".format(str(type(obj))) #warnings.warn(message)

    def export_to_file(self, filename):
        """
        Exports model to file *filename*
        """
        if len(filename.split("."))==1:
            filename += ".xml"
        xmlstring = self._dommodel.toprettyxml("  ", "\n")
        with open(filename, "w") as f:
            f.write(xmlstring)

    def _extend_dommodel(self, child):
        """
        Extends self._dommodel DOM structure with *child*
        """
        self._dommodel.childNodes[0].appendChild(child)

    @property
    def model(self):
        return self._model


########################################################################
# Code Generation Mechanism
########################################################################

INCLUDES = ["Simulation.xml", "NeuroML2CoreTypes.xml"]

class DummyCodeObject(object):
    def __init__(self, *args, **kwds):
        pass

    def __call__(self, **kwds):
        pass


class LEMSDevice(RuntimeDevice):
    '''
    The `Device` used for LEMS/NeuroML2 code genration.
    '''
    def __init__(self):
        super(LEMSDevice, self).__init__()
        self.runs = []
        self.assignments = []
        self.build_on_run = True
        self.build_options = None

    def reinit(self):
        build_on_run = self.build_on_run
        build_options = self.build_options
        self.__init__()
        super(LEMSDevice, self).reinit()
        self.build_on_run = build_on_run
        self.build_options = build_options

    # This device does not actually calculate/store any data, so the following
    # are just dummy implementations

    def synaptic_pathway_before_run(self, pathway, run_namespace):
        pass  # No spike queue initialization necessary

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
        monitors = []
        for obj in network.objects:
            one_description, one_namespace = description(obj, namespace)
            descriptions.append((obj.name, one_description))
            if type(obj) in [StateMonitor, SpikeMonitor]:
                monitors.append(obj)
            for key, value in one_namespace.iteritems():
                if key in merged_namespace and value != merged_namespace[key]:
                    raise ValueError('name "%s" is used inconsistently')
                merged_namespace[key] = value

        self.network = network
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

    def build(self, filename=None, direct_call=True):
        """
        Collecting initializers and namespace from self.runs and passing
        it to the exporter.
        """
        #if self.build_on_run and direct_call:
        #    raise RuntimeError('You used set_device with build_on_run=True '
        #                       '(the default option), which will automatically '
        #                       'build the simulation at the first encountered '
        #                       'run call - do not call device.build manually '
        #                       'in this case.')
        initializers = {}
        for descriptions, duration, namespace, assignments in self.runs:
            for assignment in assignments:
                if not assignment[2] in initializers:
                    initializers[assignment[2]] = assignment[-1]
        if len(self.runs) > 1:
            raise NotImplementedError("Currently only single run is supported.")
        if len(filename.split("."))!=1:
            filename_ = 'recording_' + filename.split(".")[0]
        else:
            filename_ = 'recording_' + filename
        exporter = NMLExporter()
        exporter.create_lems_model(self.network, namespace=namespace,
                                                 initializers=initializers,
                                                 recordingsname=filename_)
        exporter.export_to_file(filename)


lems_device = LEMSDevice()
all_devices['lemsdevice'] = lems_device
