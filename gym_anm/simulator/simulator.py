from collections import OrderedDict
import numpy as np
import scipy.optimize as optimize

from gym_anm.simulator.components import Load, TransmissionLine, \
    ClassicalGen, StorageUnit, RenewableGen, Bus, Generator
from gym_anm.constants import DEV_H
from gym_anm.simulator import check_network
from gym_anm.simulator.components.errors import PFEError


class Simulator(object):
    """
    A simulator of an AC electricity distribution network.

    Attributes
    ----------
    baseMVA : int
        The base power of the system (MVA).
    delta_t : float
        The fraction of an hour corresponding to the time interval between two
        consecutive time steps (e.g., 0.25 means an interval of 15 minutes).
    lamb : int
        The penalty factor associated with violating operating constraints, used
        in the reward signal.
    buses : dict of {int : `Bus`}
        The buses of the grid, where for each {key: value} pair, the key is a
        unique bus ID.
    branches : dict of {(int, int) : `TransmissionLine`}
        The transmission lines of the grid, where for each {key: value} pair, the
        key is a unique transmission line ID.
    devices : dict of {int : `Device`}
        The devices connected to the grid, where for each {key: value} pair, the
        key is a unique device ID.
    N_bus, N_device : int
        The number of buses and electrical devices in the network.
    N_load, N_non_slack_gen, N_des : int
        The number of load, non-slack generators, and DES devices.
    Y_bus : numpy.ndarray
        The (N_bus, N_bus) nodal admittance matrix of the network.
    specs : dict of {str : list}
        The operating characteristics of the network.
    state : dict of {str : numpy.ndarray}
        The current state of the system.

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
            The interval of time between two consecutive time steps, in fraction
            of hour.
        lamb : int
            A constant factor multiplying the penalty associated with violating
            operational constraints.
        """

        self.delta_t = delta_t
        self.lamb = lamb

        # Check the correctness of the input network file.
        check_network.check_network_specs(network)

        # Load network network.
        self.baseMVA, self.buses, self.branches, self.devices = \
            self._load_case(network)

        # Number of elements in all sets.
        self.N_bus = max([i for i in self.buses.keys()])
        self.N_device = max([d for d in self.devices.keys()])
        self.N_load = len([_ for d in self.devices.values() if isinstance(d, Load)])
        self.N_non_slack_gen = len([_ for d in self.devices.values() if isinstance(d, Generator) and not d.is_slack])
        self.N_des = len([_ for d in self.devices.values() if isinstance(d, StorageUnit)])

        # Build the nodal admittance matrix.
        self.Y_bus = self._build_admittance_matrix()

        # Compute the range of possible (P, Q) injections of each bus.
        self._compute_bus_bounds()

        # Summarize the operating range of the network.
        self.state_bounds = self.get_state_space()

        self.state = None

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
        for bus_spec in enumerate(network['bus']):
            bus = Bus(bus_spec)
            buses[bus.id] = bus

        # Create an OrderedDict ordered by bus ID.
        buses = OrderedDict(sorted(buses.items(), key=lambda t: t[0]))

        bus_ids = list(buses.keys())

        branches = {}
        for br_spec in network['branch']:
            branch = TransmissionLine(br_spec, baseMVA, bus_ids)
            branches[(branch.i_from, branch.i_to)] = branch

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
        Y_bus = np.zeros((self.N_bus, self.N_bus), dtype=np.complex)

        for (f, t), br in self.branches.items():
            # Fill an off-diagonal elements of the admittance matrix Y_bus.
            Y_bus[f, t] = - br.series / np.conjugate(br.tap)
            Y_bus[f, t] = - br.series / br.tap

            # Increment diagonal element of the admittance matrix Y_bus.
            Y_bus[f, f] += (br.series + br.shunt) \
                                         / (np.absolute(br.tap) ** 2)
            Y_bus[t, t] += br.series + br.shunt

        return Y_bus

    def _compute_bus_bounds(self):
        """
        Compute the range of (P, Q) possible injections at each bus.
        """

        P_min, P_max = {}, {}
        Q_min, Q_max = {}, {}

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

        The `init_state` vectors should have power injections in MW or MVAr and
        state of charge in MWh.

        Parameters
        ----------
        init_state : numpy.ndarray
            The initial state vector `s_0` of the environment.
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

                elif isinstance(dev, StorageUnit):
                    soc[dev_id] = soc[des_idx]
                    des_idx += 1

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
        _, _, _, _ = self.transition(P_load, P_pot, P_set_points, Q_set_points)

        # 4. Update the SoC of each DES unit to match the `initial_state`.
        for dev_id, dev in self.devices.items():
            if isinstance(dev, StorageUnit):
                dev.soc = soc[dev_id]

    # def get_network_specs(self):
    #     """
    #     Summarize the characteristics of the distribution network.
    #
    #     All values are in p.u. (or p.u. per hour).
    #
    #     Returns
    #     -------
    #     specs : dict of {str: list}
    #         A description of the operating range of the network. These values
    #         can be used to normalize observation spaces.
    #     """
    #
    #     # Bus specs.
    #     P_min_bus, P_max_bus = {}, {}
    #     Q_min_bus, Q_max_bus ={}, {}
    #     V_min_bus, V_max_bus ={}, {}
    #     for bus_id, bus in self.buses.items():
    #         P_min_bus[bus_id] = bus.p_min
    #         P_max_bus[bus_id] = bus.p_max
    #         Q_min_bus[bus_id] = bus.q_min
    #         Q_max_bus[bus_id] = bus.q_max
    #         V_min_bus[bus_id] = bus.v_min
    #         V_max_bus[bus_id] = bus.v_max
    #
    #     # Device specs.
    #     P_min_dev, P_max_dev = {}, {}
    #     Q_min_dev, Q_max_dev = {}, {}
    #     soc_min, soc_max = {}, {}
    #     for dev_id, dev in self.devices.items():
    #         P_min_dev[dev_id] = dev.p_min
    #         P_max_dev[dev_id] = dev.p_max
    #         Q_min_dev[dev_id] = dev.q_min
    #         Q_max_dev[dev_id] = dev.q_max
    #         if isinstance(dev, StorageUnit):
    #             soc_min[dev_id] = dev.soc_min
    #             soc_max[dev_id] = dev.soc_max
    #
    #     # Branch specs.
    #     S_min_br, S_max_br = {}, {}
    #     for (i, j), br in self.branches.items():
    #         S_min_br[(i, j)] = -br.rate
    #         S_max_br[(i, j)] = br.rate
    #
    #     specs = {'bus_p_min': P_min_bus, 'bus_p_max': P_max_bus,
    #              'bus_q_min': Q_min_bus, 'bus_q_max': Q_max_bus,
    #              'bus_v_magn_min': V_min_bus, 'bus_v_magn_max': V_max_bus,
    #              'dev_p_min': P_min_dev, 'dev_p_max': P_max_dev,
    #              'dev_q_min': Q_min_dev, 'dev_q_max': Q_max_dev,
    #              'soc_min': soc_min, 'soc_max': soc_max,
    #              'branch_s_max': S_max_br}
    #
    #     return specs

    def get_action_space(self):
        """
        Return the range of each possible control action.

        This function returns the lower and upper bound of the action space `\mathcal A`
        available to the agent as dictionaries. The keys are the unique device IDs
        and the values are a tuple of (upper bound, lower bound).

        Returns
        -------
        P_gen_bounds : dict of {int : tuple of float}
            The range of active (MW) power injection of generators.
        Q_gen_bounds : dict of {int : tuple of float}
            The range of reactive (MVAr) power injection of generators.
        P_des_bounds : dict of {int : tuple of float}
            The range of active (MW) power injection of DES units.
        Q_des_bounds : dict of {int : tuple of float}
            The range of reactive (MVAr) power injection of DES units.

        Notes
        -----
        The bounds returned by this function are loose, i.e., some parts of
        those spaces might be physically impossible to achieve due to other
        operating constraints. This is just an indication of the range of action
        available to the DSO. Whenever an action is taken through the transition(
        ) function, it gets mapped onto the set of physically feasible actions.
        """

        P_gen_bounds, Q_gen_bounds = {}, {}
        P_des_bounds, Q_des_bounds = {}, {}
        for dev_id, dev in self.devices.items():
            if isinstance(dev, Generator) and not dev.is_slack:
                P_gen_bounds[dev_id] = (dev.p_max * self.baseMVA, dev.p_min * self.baseMVA)
                Q_gen_bounds[dev_id] = (dev.q_max * self.baseMVA, dev.q_min * self.baseMVA)

            elif isinstance(dev, StorageUnit):
                P_des_bounds[dev_id] = (dev.p_max * self.baseMVA, dev.p_min * self.baseMVA)
                Q_des_bounds[dev_id] = (dev.q_max * self.baseMVA, dev.q_min * self.baseMVA)

        return P_gen_bounds, Q_gen_bounds, P_des_bounds, Q_des_bounds

    def get_state_space(self):
        """
        Returns the range of potential values for all state variables.

        These lower and upper bounds are respected at all timesteps in the
        simulator. For unbounded values, a range of (- inf, inf) is used.

        Returns
        -------
        dict of {str : dict}
            A dictionary where keys are the names of the state variables (e.g.,
            {'bus_p', 'bus_q', ...}) and the values are dictionary, indexed with
            the device/branch/bus unique ID, that store dictionaries of
            {units : (lower bound, upper bound)}.
        """

        # Bus bounds.
        bus_p, bus_q = {}, {}
        bus_v_magn, bus_v_ang ={}, {}
        bus_i_magn, bus_i_ang ={}, {}
        for bus_id, bus in self.buses.items():
            bus_p[bus_id] = {'MW': (bus.p_min * self.baseMVA, bus.p_max * self.baseMVA),
                             'pu': (bus.p_min, bus.p_max)}
            bus_q[bus_id] = {'MW': (bus.q_min * self.baseMVA, bus.q_max * self.baseMVA),
                             'pu': (bus.q_min, bus.q_max)}
            bus_v_magn[bus_id] = {'pu': (- np.inf, np.inf),
                                  'kV': (- np.inf, np.inf)}
            bus_v_ang[bus_id] = {'degree': (- 180, 180),
                                 'rad': (- np.pi, np.pi)}
            bus_i_magn[bus_id] = {'pu': (- np.inf, np.inf),
                                  'kA': (- np.inf, np.inf)}

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
        branch_s, branch_i_magn = {}, {}
        for br_id, branch in self.branches.items():
            branch_p[br_id] = {'MW': (- np.inf, np.inf),
                               'pu': (- np.inf, np.inf)}
            branch_q[br_id] = {'MVAr': (- np.inf, np.inf),
                               'pu': (- np.inf, np.inf)}
            branch_s[br_id] = {'MVA': (- np.inf, np.inf),
                               'pu': (- np.inf, np.inf)}
            branch_i_magn = {'pu': (- np.inf, np.inf),
                             'kA': (- np.inf, np.inf)}

        specs = {'bus_p': bus_p, 'bus_q': bus_q,
                 'bus_v_magn': bus_v_magn, 'bus_v_ang': bus_v_ang,
                 'bus_i_magn': bus_i_magn, 'bus_i_ang': bus_i_ang,
                 'dev_p': dev_p, 'dev_q': dev_q,
                 'des_soc': des_soc, 'gen_p_max': gen_p_max,
                 'branch_p': branch_p, 'branch_q': branch_q,
                 'branch_s': branch_s, 'branch_i_magn': branch_i_magn
                }

        return specs

    def transition(self, P_load, P_potential, P_set_points, Q_set_points):
        """
        Simulate a transition of the system from time t to time (t+1).

        This function simulates a transition of the system after actions were
        taken by the DSO. The results of these decisions then affect the new
        state of the system, and the associated reward is returned.

        Parameters
        ----------
        P_load : dict of {int : float}
            A dictionary with device IDs as keys and fixed real power injection
            (MW) as values (load devices only).
        P_potential : dict of {int : float}
            A dictionary with device IDs as keys and maximum potential real power
            generation (MW) as values (generators only).
        P_set_points : dict of {int : float}
            A dictionary with device IDs as keys and real power injection
            set-points (MW) set by the DSO (non-slack generators and DES units).
        Q_set_points : dict of {int : float}
            A dictionary with device IDs as keys and reactive power injection
            set-points (MVAr) set by the DSO (non-slack generators and DES units).

        Returns
        -------
        state : dict of {str : numpy.ndarray}
            The current state of the system.
        reward : float
            The reward associated with the transition.
        e_loss : float
            The total energy loss (MWh).
        penalty : float
            The total penalty due to violation of operating constraints.
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

        # 4c. Solve the network equations and compute nodal V (p.u.), P (MW),
        # and Q (MVAr) vectors.
        self._solve_pfes()

        # 5a. Compute branch I (p.u.), P (MW) and Q (MVAR) flows.
        for branch in self.branches.values():
            v_f = self.buses[branch.i_from]
            v_t = self.buses[branch.i_to]
            branch.compute_currents(v_f, v_t)
            branch.compute_power_flows(v_f, v_t)

        # 6. Construct the new state of the network.
        self.state = self._gather_state()

        # 7. Compute the reward associated with the transition.
        reward, e_loss, penalty = self._compute_reward()

        return self.state, reward, e_loss, penalty

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

    def _solve_pfes(self):
        """
        Solve the power flow equations and return V, P, Q for each bus.

        Raises
        ------
        ValueError
            Raised if no solution is found.
        """

        P_bus, Q_bus = np.zeros(self.N_bus), np.zeros(self.N_bus)
        V_init = np.array([1.] * self.N_bus + [0.] * self.N_bus)
        v_slack, slack_id = None, None

        for bus_id, bus in self.buses.items():
            # Fix active power injections, excluding slack bus.
            if not bus.is_slack:
                P_bus[bus_id] = bus.p
                Q_bus[bus_id] = bus.q

            # Fix slack bus voltage to (v_slack * exp(j0)).
            if bus.is_slack:
                v_slack = bus.v_slack
                slack_id = bus_id

        # Solve the power flow equations of the network.
        sol = optimize.root(self._power_flow_eqs, V_init,
                          args=(self.Y_bus, P_bus, Q_bus, slack_id, v_slack),
                          method='lm', options={'xtol': 1.0e-4})
        if not sol.success:
            raise PFEError('No solution to the PFEs: ', sol.message)
        x = sol.x

        # Re-create the complex bus voltage vector.
        V = x[0:self.N_bus] + 1.j * x[self.N_bus:]

        # Compute nodal current injections.
        I = np.dot(self.Y_bus, V)

        for bus_id, bus in self.buses.items():
            # Update the complex bus voltage and current injections.
            bus.v = V[bus_id]
            bus.i = I[bus_id]

            # Compute the power injections at the slack bus.
            if bus.is_slack:
                s = bus.v * np.conjugate(bus.i)
                bus.p = s.real
                bus.q = s.imag

        return

    def _power_flow_eqs(self, v, Y, P, Q, slack_id, v_slack):
        """
        Return the power flow equations to be solved.

        Parameters
        ----------
        v : 1D numpy.ndarray
            Initial bus voltage guess, where elements 0 to N-1 represent the
            real part of the initial guess for nodal voltage, and the N to 2N-1
            elements their corresponding imaginary parts (p.u.).
        Y : 2D numpy.ndarray
            The network bus admittance matrix (p.u.).
        P : 1D numpy.ndarray
            Fixed real power nodal injections (p.u.).
        Q : 1D numpy.ndarray
            Fixed reactive power nodal injections (p.u.).
        slack_id : int
            The unique bus ID of the slack bus.
        v_slack : float
            The fixed voltage magnitude at the slack bus (p.u.).

        Returns
        -------
        2D numpy.ndarray
            A vector of 4 expressions to find the roots of.
        """

        # Re-build complex nodal voltage array.
        V = v[:self.N_bus] + 1.j * v[self.N_bus:]

        # Create complex equations as a matrix product.
        complex_rhs = V * np.conjugate(np.dot(Y, V))

        # Equations involving real variables and real power injections (skip slack bus).
        real_p = np.real(np.delete(complex_rhs, slack_id)) - np.delete(P, slack_id)

        # Equations involving real variables and reactive power injections (skip slack bus).
        real_q = np.imag(np.delete(complex_rhs, slack_id)) - np.delete(Q, slack_id)

        # Equation fixing the voltage magnitude at the slack bus.
        slack_magn = [np.absolute(V[slack_id]) - v_slack]

        # Equation fixing the voltage phase angle to be 0.
        slack_phase = [np.angle(V[slack_id])]

        # Stack equations made of only real variables on top of each other.
        real_equations = np.hstack((real_p, real_q, slack_magn, slack_phase))

        return real_equations

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
        branch_s, branch_i_magn = {'pu': {}, 'MVA': {}}, {'pu': {}}
        for (f, t), branch in self.branches.items():
            branch_p['pu'][(f, t)] = np.sign(branch.p_from) * np.maximum(np.abs(branch.p_from), np.abs(branch.p_to))
            branch_p['MW'][(f, t)] = branch_p['pu'][(f, t)] * self.baseMVA

            branch_q['pu'][(f, t)] = np.sign(branch.q_from) * np.maximum(np.abs(branch.q_from), np.abs(branch.q_to))
            branch_q['MVAr'][(f, t)] = branch_q['pu'][(f, t)] * self.baseMVA

            branch_s['pu'][(f, t)] = branch.s_apparent_max
            branch_s['MVA'][(f, t)] = branch.s_apparent_max * self.baseMVA

            branch_i_magn['pu'][(f, t)] = np.sign(branch.i_from) * np.maximum(np.abs(branch.i_from), np.abs(branch.i_to))

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
                 'branch_i_magn': branch_i_magn
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
            elif isinstance(dev, StorageUnit):
                e_loss -= dev.p
            else:
                raise NotImplementedError

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
            penalty += np.maximum(0, branch.s_apparent_max - branch.rate)

        penalty *= self.delta_t

        # Compute the total reward.
        reward = - (e_loss + self.lamb * penalty)

        return reward, e_loss, penalty
