from collections import OrderedDict
import numpy as np
import copy
from scipy.sparse import csc_matrix
from logging import getLogger

from .components import Load, TransmissionLine, \
    ClassicalGen, StorageUnit, RenewableGen, Bus, Generator
from .components.constants import DEV_H
from . import check_network
from . import solve_load_flow

logger = getLogger(__file__)


class Simulator(object):
    """
    A simulator of an AC electricity distribution network.

    Attributes
    ----------
    baseMVA : int
        The base power of the system (MVA).
    delta_t : float
        The fraction of an hour corresponding to the time interval between two
        consecutive time steps :math:`\\Delta t` (e.g., 0.25 means an interval of 15 minutes).
    lamb : int
        The penalty factor associated with violating operating constraints :math:`\\lambda`,
        used in the reward signal.
    buses : dict of {int : :py:class:`Bus`}
        The buses of the grid, where for each {key: value} pair, the key is a
        unique bus ID.
    branches : dict of {(int, int) : `TransmissionLine`}
        The transmission lines of the grid, where for each {key: value} pair, the
        key is a unique transmission line ID.
    devices : dict of {int : `Device`}
        The devices connected to the grid, where for each {key: value} pair, the
        key is a unique device ID.
    N_bus, N_device : int
        The number of buses :math:`|\\mathcal N|` and electrical devices :math:`|\\mathcal D|`
        in the network.
    N_load, N_non_slack_gen, N_des, N_gen_rer : int
        The number of load :math:`|\\mathcal D_L|`, non-slack generators :math:`|\\mathcal D_G|-1`,
        DES devices :math:`|\\mathcal D_{DES}|`, and renewable energy generators :math:`|\\mathcal D_{DER}|`.
    Y_bus : scipy.sparse.csc_matrix
        The (N_bus, N_bus) nodal admittance matrix :math:`\\mathbf Y` of the network as a sparse
        matrix.
    state_bounds : dict of {str : dict}
        The lower and upper bounds that each electrical quantity may take. This
        is a nested dictionary with keys [quantity][ID][unit], where quantity
        is the electrical quantity of interest, ID is the unique bus/device/branch
        ID and unit is the units in which to return the quantity.
    state : dict of {str : numpy.ndarray}
        The current state of the system.
    pfe_converged : bool
        True if the load flow converged; otherwise False (possibly infeasible
        problem).

    Methods
    -------
    reset(np_random, init_soc)
        Reset the simulator.
    get_network_specs()
        Get the operating characteristics of the network.
    get_action_space()
        Get the control action space available.
    transition(P_load, P_potential, P_curt_limit, desired_alpha, Q_storage)
        Simulate a transition of the system from time t to time (t+1).
    """

    def __init__(self, network, delta_t, lamb):
        """
        Parameters
        ----------
        network : dict of {str : numpy.ndarray}
            A network dictionary describing the power grid.
        delta_t : float
            The interval of time between two consecutive time steps :math:`\\delta t`,
            in fraction of hour.
        lamb : int
            A constant factor :math:`\\lambda` multiplying the penalty associated
            with violating operational constraints.
        """

        self.delta_t = delta_t
        self.lamb = lamb

        # Check the correctness of the input network file.
        check_network.check_network_specs(network)

        # Load network.
        self.baseMVA, self.buses, self.branches, self.devices = \
            self._load_case(network)

        # Number of elements in all sets.
        self.N_bus = len(self.buses)
        self.N_device = len(self.devices)
        self.N_load = len([0 for d in self.devices.values() if isinstance(d, Load)])
        self.N_non_slack_gen = len([0 for d in self.devices.values() if isinstance(d, Generator) and not d.is_slack])
        self.N_des = len([0 for d in self.devices.values() if isinstance(d, StorageUnit)])
        self.N_gen_rer = len([0 for d in self.devices.values() if isinstance(d, RenewableGen)])

        # Build the nodal admittance matrix.
        self.Y_bus = self._build_admittance_matrix()

        # Compute the range of possible (P, Q) injections of each bus.
        self._compute_bus_bounds()

        # Summarize the operating range of the network.
        self.state_bounds = self.get_state_space()

        self.state = None
        self.pfe_converged = None

    def _load_case(self, network):
        """
        Initialize the network model based on parameters given in a network file.

        Parameters
        ----------
        network : dict of numpy.ndarray
            A network dictionary describing the power grid.

        Returns
        -------
        baseMVA : int or float
            The base power of the system (MVA).
        buses : OrderedDict of {int : `Bus`}
            The buses of the network, ordered by unique bus ID.
        branches : dict of {(int, int) : `TransmissionLine`}
            The branches of the network.
        devices : OrderedDict of {int : `Device`}
            The electrical devices connected to the grid, ordered by unique device
            ID.

        Raises
        ------
        NoImplementedError
            When the feature DEV_TYPE of a device is not valid.
        """

        baseMVA = network['baseMVA']

        buses = {}
        for bus_spec in network['bus']:
            bus = Bus(bus_spec)
            buses[bus.id] = bus

        # Create an OrderedDict ordered by bus ID.
        buses = OrderedDict(sorted(buses.items(), key=lambda t: t[0]))

        bus_ids = list(buses.keys())

        branches = OrderedDict()  # order branches in the order they are provided.
        for br_spec in network['branch']:
            branch = TransmissionLine(br_spec, baseMVA, bus_ids)
            branches[(branch.f_bus, branch.t_bus)] = branch

        devices = {}
        for dev_spec in network['device']:
            dev_type = int(dev_spec[DEV_H['DEV_TYPE']])

            if dev_type == -1:
                dev = Load(dev_spec, bus_ids, baseMVA)

            elif dev_type in [0, 1]:
                dev = ClassicalGen(dev_spec, bus_ids, baseMVA)

            elif dev_type == 2:
                dev = RenewableGen(dev_spec, bus_ids, baseMVA)

            elif dev_type == 3:
                dev = StorageUnit(dev_spec, bus_ids, baseMVA)

            else:
                raise NotImplementedError

            devices[dev.dev_id] = dev

        # Create an OrderedDict sorted by device ID.
        devices = OrderedDict(sorted(devices.items(), key=lambda t: t[0]))

        return baseMVA, buses, branches, devices

    def _build_admittance_matrix(self):
        """
        Build the nodal admittance matrix of the network (in p.u.).
        """
        n = max([i for i in self.buses.keys()])
        Y_bus = np.zeros((n + 1, n + 1), dtype=np.complex)

        for (f, t), br in self.branches.items():
            # Fill an off-diagonal elements of the admittance matrix Y_bus.
            Y_bus[f, t] = - br.series / np.conjugate(br.tap)
            Y_bus[t, f] = - br.series / br.tap

            # Increment diagonal element of the admittance matrix Y_bus.
            Y_bus[f, f] += (br.series + br.shunt) / (np.abs(br.tap) ** 2)
            Y_bus[t, t] += br.series + br.shunt

        return csc_matrix(Y_bus)

    def _compute_bus_bounds(self):
        """
        Compute the range of (P, Q) possible injections at each bus.
        """

        P_min = {i:0 for i in self.buses.keys()}
        P_max = copy.copy(P_min)
        Q_min, Q_max = copy.copy(P_min), copy.copy(P_min)

        # Iterate over all devices connected to the power grid.
        for dev in self.devices.values():
            P_min[dev.bus_id] += dev.p_min
            P_max[dev.bus_id] += dev.p_max
            Q_min[dev.bus_id] += dev.q_min
            Q_max[dev.bus_id] += dev.q_max

        # Update each bus with its operation range.
        for bus_id, bus in self.buses.items():
            if bus_id in P_min.keys():
                bus.p_min = P_min[bus_id]
                bus.p_max = P_max[bus_id]
                bus.q_min = Q_min[bus_id]
                bus.q_max = Q_max[bus_id]

    def reset(self, init_state):
        """
        Reset the simulator.

        The :code:`init_state` vector should have power injections in MW or MVAr and
        state of charge in MWh.

        Parameters
        ----------
        init_state : numpy.ndarray
            The initial state vector :math:`s_0` of the environment.

        Returns
        -------
        pfe_converged : bool
            True if a feasible solution was reached (within the specified
            tolerance) for the power flow equations. If False, it might indicate
            that the network has collapsed (e.g., voltage collapse).
        """

        self.state = None

        # 1. Extract variables to pass to `transition` function.
        P_dev = init_state[:self.N_device]
        Q_dev = init_state[self.N_device: 2 * self.N_device]
        soc = init_state[2 * self.N_device: 2 * self.N_device + self.N_des]
        P_max = init_state[2 * self.N_device + self.N_des: 2 * self.N_device + self.N_des + self.N_non_slack_gen]

        P_load = {}
        P_pot = {}
        P_set_points, Q_set_points = {}, {}
        gen_idx, des_idx = 0, 0

        for idx, (dev_id, dev) in enumerate(self.devices.items()):
            if isinstance(dev, Load):
                P_load[dev_id] = P_dev[idx]

            elif isinstance(dev, (Generator, StorageUnit)) and not dev.is_slack:
                P_set_points[dev_id] = P_dev[idx]
                Q_set_points[dev_id] = Q_dev[idx]

                if isinstance(dev, Generator) and not dev.is_slack:
                    P_pot[dev_id] = P_max[gen_idx]
                    gen_idx += 1

        # 2. Set the initial SoC of each DES unit to either empty or full, so
        # that the corresponding power injection (given in `init_state`) will
        # be made possible by the simulator during the `transition` call.
        for dev_id, dev in self.devices.items():
            if isinstance(dev, StorageUnit):
                if P_set_points[dev_id] <= 0:
                    dev.soc = dev.soc_min
                else:
                    dev.soc = dev.soc_max

        # 3. Compute all electrical quantities in the network.
        _, _, _, _, pfe_converged = \
            self.transition(P_load, P_pot, P_set_points, Q_set_points)

        # 4. Update the SoC of each DES unit to match the `initial_state`.
        soc_idx = 0
        for dev in self.devices.values():
            if isinstance(dev, StorageUnit):
                dev.soc = soc[soc_idx] / self.baseMVA
                soc_idx += 1

        # 5. Re-construct the state dictionary after modifying the SoC.
        self.state = self._gather_state()

        return pfe_converged

    def get_rendering_specs(self):
        """
        Summarize the specs of the distribution network (useful for rendering).

        All values are in p.u. (or p.u. per hour for DES units). This method
        differs from :code:`get_state_space()` in that it returns the bounds that
        should be respected for a successful operation of the network, whereas
        :code:`get_state_space()` returns the range of values that state variables can
        take. For example, this method will return the rate of a branch, whereas
        :code:`get_state_space()` will return an upper bound of :code:`np.inf`, since branch
        rates may be violated during the simulation.

        Returns
        -------
        specs : dict of {str : dict}
            A dictionary where keys are the names of the state variables (e.g.,
            {'bus_p', 'bus_q', ...}) and the values are dictionary, indexed with
            the device/branch/bus unique ID, that store dictionaries of
            {units : (lower bound, upper bound)}.
        """

        dev_type = {}
        for dev_id, dev in self.devices.items():
            dev_type[dev_id] = dev.type

        v_bus = {}
        for bus_id, bus in self.buses.items():
            v_bus[bus_id] = {'pu': (bus.v_min, bus.v_max),
                             'kV': (bus.v_min * bus.baseKV, bus.v_max * bus.baseKV)}

        branch_s = {}
        for branch_id, branch in self.branches.items():
            branch_s[branch_id] = {'MVA': (0, branch.rate * self.baseMVA),
                                   'pu': (0, branch.rate)}

        specs = {'bus_p': self.state_bounds['bus_p'],
                 'bus_q': self.state_bounds['bus_q'],
                 'dev_p': self.state_bounds['dev_p'],
                 'dev_q': self.state_bounds['dev_q'],
                 'bus_v': v_bus, 'dev_type': dev_type,
                 'des_soc': self.state_bounds['des_soc'],
                 'branch_s': branch_s}

        return specs

    def get_action_space(self):
        """
        Return the range of each possible control action.

        This function returns the lower and upper bound of the action space :math:`\\mathcal A`
        available to the agent as dictionaries. The keys are the unique device IDs
        and the values are a tuple of (lower bound, upper bound).

        Returns
        -------
        P_gen_bounds : dict of {int : tuple of float}
            The range of active (MW) power injection of generators :math:`[\\overline P_g, \\underline P_g]`.
        Q_gen_bounds : dict of {int : tuple of float}
            The range of reactive (MVAr) power injection of generators :math:`[\\overline Q_g, \\underline Q_g]`.
        P_des_bounds : dict of {int : tuple of float}
            The range of active (MW) power injection of DES units :math:`[\\overline P_d, \\underline P_d]`.
        Q_des_bounds : dict of {int : tuple of float}
            The range of reactive (MVAr) power injection of DES units :math:`[\\overline Q_d, \\underline Q_d]`.

        Notes
        -----
        The bounds returned by this function are loose, i.e., some parts of
        those spaces might be physically impossible to achieve due to other
        operating constraints. This is just an indication of the range of action
        available to the DSO. Whenever an action is taken through the :code:`transition()`
        function, it gets mapped onto the set of physically feasible actions.
        """

        P_gen_bounds, Q_gen_bounds = {}, {}
        P_des_bounds, Q_des_bounds = {}, {}
        for dev_id, dev in self.devices.items():
            if isinstance(dev, Generator) and not dev.is_slack:
                P_gen_bounds[dev_id] = (dev.p_min * self.baseMVA, dev.p_max * self.baseMVA)
                Q_gen_bounds[dev_id] = (dev.q_min * self.baseMVA, dev.q_max * self.baseMVA)

            elif isinstance(dev, StorageUnit):
                P_des_bounds[dev_id] = (dev.p_min * self.baseMVA, dev.p_max * self.baseMVA)
                Q_des_bounds[dev_id] = (dev.q_min * self.baseMVA, dev.q_max * self.baseMVA)

        return P_gen_bounds, Q_gen_bounds, P_des_bounds, Q_des_bounds

    def get_state_space(self):
        """
        Returns the range of potential values for all state variables.

        These lower and upper bounds are respected at all timesteps in the
        simulator. For unbounded values, a range of :code:`(-inf, inf)` is used.

        Returns
        -------
        specs : dict of {str : dict}
            A dictionary where keys are the names of the state variables (e.g.,
            {'bus_p', 'bus_q', ...}) and the values are dictionary, indexed with
            the device/branch/bus unique ID, that store dictionaries of
            {units : (lower bound, upper bound)}.
        """

        # Bus bounds.
        bus_p, bus_q = {}, {}
        bus_v_magn, bus_v_ang = {}, {}
        bus_i_magn, bus_i_ang = {}, {}
        for bus_id, bus in self.buses.items():
            bus_p[bus_id] = {'MW': (bus.p_min * self.baseMVA, bus.p_max * self.baseMVA),
                             'pu': (bus.p_min, bus.p_max)}
            bus_q[bus_id] = {'MVAr': (bus.q_min * self.baseMVA, bus.q_max * self.baseMVA),
                             'pu': (bus.q_min, bus.q_max)}
            if bus.is_slack:
                bus_v_magn[bus_id] = {'pu': (bus.v_slack, bus.v_slack),
                                      'kV': (bus.v_slack * bus.baseKV, bus.v_slack * bus.baseKV)}
                bus_v_ang[bus_id] = {'degree': (0, 0),
                                     'rad': (0, 0)}
            else:
                bus_v_magn[bus_id] = {'pu': (- np.inf, np.inf),
                                      'kV': (- np.inf, np.inf)}
                bus_v_ang[bus_id] = {'degree': (- 180, 180),
                                     'rad': (- np.pi, np.pi)}
            bus_i_magn[bus_id] = {'pu': (- np.inf, np.inf),
                                  'kA': (- np.inf, np.inf)}
            bus_i_ang[bus_id] = {'degree': (- 180, 180),
                                 'rad': (- np.pi, np.pi)}

        # Device bounds.
        dev_p, dev_q = {}, {}
        des_soc, gen_p_max = {}, {}
        for dev_id, dev in self.devices.items():
            dev_p[dev_id] = {'MW': (dev.p_min * self.baseMVA, dev.p_max * self.baseMVA),
                             'pu': (dev.p_min, dev.p_max)}
            dev_q[dev_id] = {'MVAr': (dev.q_min * self.baseMVA, dev.q_max * self.baseMVA),
                             'pu': (dev.q_min, dev.q_max)}
            if isinstance(dev, StorageUnit):
                des_soc[dev_id] = {'MWh': (dev.soc_min * self.baseMVA, dev.soc_max * self.baseMVA),
                                   'pu': (dev.soc_min, dev.soc_max)}
            if isinstance(dev, Generator) and not dev.is_slack:
                gen_p_max[dev_id] = {'MW': (dev.p_min * self.baseMVA, dev.q_max * self.baseMVA),
                                     'pu': (dev.p_min, dev.p_max)}

        # Branch bounds.
        branch_p, branch_q = {}, {}
        branch_s, branch_i_magn, branch_i_ang = {}, {}, {}
        for br_id, branch in self.branches.items():
            branch_p[br_id] = {'MW': (- np.inf, np.inf),
                               'pu': (- np.inf, np.inf)}
            branch_q[br_id] = {'MVAr': (- np.inf, np.inf),
                               'pu': (- np.inf, np.inf)}
            branch_s[br_id] = {'MVA': (- np.inf, np.inf),
                               'pu': (- np.inf, np.inf)}
            branch_i_magn[br_id] = {'pu': (- np.inf, np.inf),
                                    'kA': (- np.inf, np.inf)}
            branch_i_ang[br_id] = {'rad': (- np.pi, np.pi),
                                   'degree': (- 180, 180)}

        specs = {'bus_p': bus_p, 'bus_q': bus_q,
                 'bus_v_magn': bus_v_magn, 'bus_v_ang': bus_v_ang,
                 'bus_i_magn': bus_i_magn, 'bus_i_ang': bus_i_ang,
                 'dev_p': dev_p, 'dev_q': dev_q,
                 'des_soc': des_soc, 'gen_p_max': gen_p_max,
                 'branch_p': branch_p, 'branch_q': branch_q,
                 'branch_s': branch_s,
                 'branch_i_magn': branch_i_magn, 'branch_i_ang': branch_i_ang
                }

        return specs

    def transition(self, P_load, P_potential, P_set_points, Q_set_points):
        """
        Simulate a transition of the system from time :math:`t` to time :math:`t+1`.

        This function simulates a transition of the system after actions were
        taken by the DSO. The results of these decisions then affect the new
        state of the system, and the associated reward is returned.

        Parameters
        ----------
        P_load : dict of {int : float}
            A dictionary with device IDs as keys and fixed real power injection :math:`P_l^{(dev)}`
            (MW) as values (load devices only).
        P_potential : dict of {int : float}
            A dictionary with device IDs as keys and maximum potential real power
            generation :math:`P_g^{(max)}` (MW) as values (generators only).
        P_set_points : dict of {int : float}
            A dictionary with device IDs as keys and real power injection
            set-points :math:`a_P` (MW) set by the DSO (non-slack generators and DES units).
        Q_set_points : dict of {int : float}
            A dictionary with device IDs as keys and reactive power injection
            set-points :math:`a_Q` (MVAr) set by the DSO (non-slack generators and DES units).

        Returns
        -------
        state : dict of {str : numpy.ndarray}
            The new state of the system :math:`s_{t+1}`.
        reward : float
            The reward :math:`r_t` associated with the transition.
        e_loss : float
            The total energy loss :math:`\\Delta E_{t:t+1}` (MWh).
        penalty : float
            The total penalty :math:`\\lambda \\phi(s_{t+1})` due to violation of operating constraints.
        pfe_converged : bool
            True if a feasible solution was reached (within the specified
            tolerance) for the power flow equations. If False, it might indicate
            that the network has collapsed (e.g., voltage collapse).
        """

        for dev_id, dev in self.devices.items():

            # 1. Compute the (P, Q) injection point of each load.
            if isinstance(dev, Load):
                dev.map_pq(P_load[dev_id] / self.baseMVA)

            # 2. Compute the (P, Q) injection point of each non-slack generator.
            elif isinstance(dev, Generator) and not dev.is_slack:
                dev.p_pot = np.clip(P_potential[dev_id] / self.baseMVA, dev.p_min, dev.p_max)
                dev.map_pq(P_set_points[dev_id] / self.baseMVA,
                           Q_set_points[dev_id] / self.baseMVA)

            # 3. Compute the (P, Q) injection point of each DES unit and update the
            # new SoC.
            elif isinstance(dev, StorageUnit):
                dev.map_pq(P_set_points[dev_id] / self.baseMVA,
                           Q_set_points[dev_id] / self.baseMVA,
                           self.delta_t)
                dev.update_soc(self.delta_t)

            # 4a. Initialize the (P, Q) injection point of the slack bus device to 0.
            elif dev.is_slack:
                dev.p = 0.
                dev.q = 0.

        # 4b. Compute the total (P, Q) injection at each bus.
        self._get_bus_total_injections()

        # 5. Solve the network equations and compute nodal V, P, and Q vectors.
        _, self.pfe_converged = \
            solve_load_flow.solve_pfe_newton_raphson(self, xtol=1e-5)

        # 6. Construct the new state of the network.
        self.state = self._gather_state()

        # 7. Compute the reward associated with the transition.
        reward, e_loss, penalty = self._compute_reward()

        return self.state, reward, e_loss, penalty, self.pfe_converged

    def _get_bus_total_injections(self):
        """
        Compute the total (P, Q) injection point at each bus.
        """
        for bus in self.buses.values():
            bus.p = 0.
            bus.q = 0.

        for dev in self.devices.values():
            self.buses[dev.bus_id].p += dev.p
            self.buses[dev.bus_id].q += dev.q

    def _gather_state(self):
        """
        Gather all electrical quantities of the network in a single dictionary.

        All values are gathered in all supported units.
        """

        # Collect bus variables.
        bus_p, bus_q = {'pu': {}, 'MW': {}}, {'pu': {}, 'MVAr': {}}
        bus_v_magn, bus_v_ang = {'pu': {}, 'kV': {}}, {'rad': {}, 'degree': {}}
        bus_i_magn, bus_i_ang = {'pu': {}, 'kA': {}}, {'rad': {}, 'degree': {}}
        for bus_id, bus in self.buses.items():
            bus_p['pu'][bus_id] = bus.p
            bus_p['MW'][bus_id] = bus.p * self.baseMVA

            bus_q['pu'][bus_id] = bus.q
            bus_q['MVAr'][bus_id] = bus.q * self.baseMVA

            bus_v_magn['pu'][bus_id] = np.abs(bus.v)
            bus_v_magn['kV'][bus_id] = np.abs(bus.v) * bus.baseKV

            bus_v_ang['rad'][bus_id] = np.angle(bus.v)
            bus_v_ang['degree'][bus_id] = np.angle(bus.v) * 180 / np.pi

            bus_i_magn['pu'][bus_id] = np.abs(bus.i)
            bus_i_magn['kA'][bus_id] = np.abs(bus.i) * self.baseMVA / bus.baseKV

            bus_i_ang['rad'][bus_id] = np.angle(bus.i)
            bus_i_ang['degree'][bus_id] = np.angle(bus.i) * 180 / np.pi

        # Collect device variables.
        dev_p, dev_q = {'pu': {}, 'MW': {}}, {'pu': {}, 'MVAr': {}}
        des_soc, gen_p_max = {'pu': {}, 'MWh': {}}, {'pu': {}, 'MW': {}}
        for dev_id, dev in self.devices.items():
            dev_p['pu'][dev_id] = dev.p
            dev_p['MW'][dev_id] = dev.p * self.baseMVA

            dev_q['pu'][dev_id] = dev.q
            dev_q['MVAr'][dev_id] = dev.q * self.baseMVA

            if isinstance(dev, StorageUnit):
                des_soc['pu'][dev_id] = dev.soc
                des_soc['MWh'][dev_id] = dev.soc * self.baseMVA

            if isinstance(dev, Generator) and not dev.is_slack:
                gen_p_max['pu'][dev_id] = dev.p_pot
                gen_p_max['MW'][dev_id] = dev.p_pot * self.baseMVA

        # Collect branch variables.
        branch_p, branch_q = {'pu': {}, 'MW': {}}, {'pu': {}, 'MVAr': {}}
        branch_s = {'pu': {}, 'MVA': {}}
        branch_i_magn, branch_i_ang = {'pu': {}}, {'rad': {}, 'degree': {}}
        for (f, t), branch in self.branches.items():
            branch_p['pu'][(f, t)] = branch.p_from
            branch_p['MW'][(f, t)] = branch.p_from * self.baseMVA

            branch_q['pu'][(f, t)] = branch.q_from
            branch_q['MVAr'][(f, t)] = branch.q_from * self.baseMVA

            branch_s['pu'][(f, t)] = branch.s_apparent_max
            branch_s['MVA'][(f, t)] = branch.s_apparent_max * self.baseMVA

            branch_i_magn['pu'][(f, t)] = np.sign(branch.i_from).real * np.abs(branch.i_from)

            branch_i_ang['rad'][(f, t)] = np.angle(branch.i_from)
            branch_i_ang['degree'][(f, t)] = np.angle(branch.i_from) * 180 / np.pi

        state = {'bus_p': bus_p,
                 'bus_q': bus_q,
                 'bus_v_magn': bus_v_magn,
                 'bus_v_ang': bus_v_ang,
                 'bus_i_magn': bus_i_magn,
                 'bus_i_ang': bus_i_ang,
                 'dev_p': dev_p,
                 'dev_q': dev_q,
                 'des_soc': des_soc,
                 'gen_p_max': gen_p_max,
                 'branch_p': branch_p,
                 'branch_q': branch_q,
                 'branch_s': branch_s,
                 'branch_i_magn': branch_i_magn,
                 'branch_i_ang': branch_i_ang
                 }

        return state

    def _compute_reward(self):
        """
        Return the total reward associated with the current state of the system.

        The reward is computed as a negative sum of transmission
        losses, curtailment losses, (dis)charging losses, and operational
        constraints violation costs.

        Returns
        -------
        reward : float
            The total reward associated with the transition to a new system
            state.
        e_loss : float
            The total energy loss (p.u. per hour).
        penalty : float
            The total penalty due to violation of operating constraints (p.u. per
            hour).
        """

        # Compute the energy loss.
        e_loss = 0.
        for dev in self.devices.values():
            if isinstance(dev, (Generator, Load)):
                e_loss += dev.p

            if isinstance(dev, RenewableGen):
                e_loss += np.maximum(0, dev.p_pot - dev.p)

        e_loss *= self.delta_t

        # Compute the penalty term.
        penalty = 0.
        for bus in self.buses.values():
            v_magn = np.abs(bus.v)
            penalty += np.maximum(0, v_magn - bus.v_max) \
                       + np.maximum(0, bus.v_min - v_magn)

        for branch in self.branches.values():
            penalty += np.maximum(0, np.abs(branch.s_apparent_max) - branch.rate)

        penalty *= self.delta_t * self.lamb

        # Compute the total reward.
        reward = - (e_loss + penalty)

        return reward, e_loss, penalty
