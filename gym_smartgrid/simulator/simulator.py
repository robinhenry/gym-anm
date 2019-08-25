import numpy as np
import scipy.optimize as optimize

from gym_smartgrid.simulator.components import Load, TransmissionLine, \
    PowerPlant, Storage, VRE, Bus
from gym_smartgrid.constants import DEV_H, BRANCH_H


class Simulator(object):
    """
    A simulator of a single-phase AC electricity distribution network.

    Attributes
    ----------
    rng : numpy.random.RandomState
        The random  seed.
    time_factor : float
        The fraction of an hour corresponding to the time interval between two
        consecutive time steps, e.g. 0.25 means an interval of 15 minutes.
    lamb : int
        A penalty factor associated with violating operating constraints.
    baseMVA : int
        The base power of the system (MVA).
    buses : dict of {int : `Bus`}
        The buses of the grid, where for each {key: value} pair, the key is a
        unique bus ID.
    branches : dict of {int : `TransmissionLine`}
        The transmission lines of the grid, where for each {key: value} pair, the
        key is a unique transmission line ID.
    gens : dict of {int : `Generator`}
        The generators of the grid, where for each {key: value} pair, the key is
        a unique generator ID. The slack bus is *not* included.
    storages : dict of {int : `Storage`}
        The storage units of the grid, where for each {key: value} pair, the key
        is a unique storage unit ID.
    slack_dev : `Generator`
        The single generator connected to the slack bus.
    N_bus, N_branch, N_load, N_device, N_storage : int
        The number of elements in each set (including slack bus and device).
    Y_bus : 2D numpy.ndarray
        The nodal admittance matrix of the network.
    specs : dict of {str : list}
        The operating characteristics of the network.
    state : dict of {str : array_like}
        The current state of the system.

    Methods
    -------
    reset()
        Reset the simulator.
    get_network_specs()
        Get the operating characteristics of the network.
    get_action_space()
        Get the control action space available.
    transition(P_load, P_potential, P_curt_limit, desired_alpha, Q_storage)
        Simulate a transition of the system from time t to time (t+1).
    """

    def __init__(self, case, delta_t=15, lamb=1e3, rng=None):
        """
        Parameters
        ----------
        case : dict of array_like
            A case dictionary describing the power grid.
        delta_t : int, optional
            The interval of time between two consecutive time steps (in minutes).
        lamb : int, optional
            A constant factor multiplying the penalty associated with violating
            operational constraints.
        rng : np.random.RandomState, optional
            A random seed.
        """

        # Check the correctness of the input case file.
        #utils.check_casefile(case)

        self.time_factor = delta_t / 60.
        self.lamb = lamb

        # Initialize random generator.
        self.rng = np.random.RandomState() if rng is None else rng

        # Load network case.
        self._load_case(case)

        # Number of elements in all sets.
        self.N_bus = len(self.buses)
        self.N_branch = len(self.branches)
        self.N_load = len(self.loads)
        self.N_gen = len(self.gens) + 1  # +1 for slack bus.
        self.N_storage = len(self.storages)
        self.N_device = self.N_gen + self.N_load + self.N_storage

        # Build the nodal admittance matrix.
        self.Y_bus = self._build_admittance_matrix()

        # Compute the range of possible (P, Q) injections of each bus.
        self._compute_bus_bounds()

        # Summarize the operating range of the network.
        self.specs = self.get_network_specs()

    def _load_case(self, case):
        """
        Initialize the network model based on parameters given in a case file.
        """

        self.baseMVA = case['baseMVA']

        self.buses = []
        for bus_id, bus in enumerate(case['bus']):
            self.buses.append(Bus(bus))

        self.branches = []
        for br in case['branch']:
            if br[BRANCH_H['BR_STATUS']]:
                self.branches.append(TransmissionLine(br, self.baseMVA))

        self.loads, self.gens, self.storages = {}, {}, {}
        dev_idx, load_idx, gen_idx, su_idx = 0, 0, 0, 0
        for dev in case['device']:
            if dev[DEV_H['DEV_STATUS']]:
                dev_type = int(dev[DEV_H['DEV_TYPE']])

                if dev_type == -1:
                    self.loads[dev_idx] = Load(dev_idx, load_idx, dev)
                    load_idx += 1

                elif dev_type == 0:
                    self.slack_dev = PowerPlant(dev_idx, 0, dev)

                elif dev_type == 1:
                    self.gens[dev_idx] = PowerPlant(dev_idx, gen_idx, dev)
                    gen_idx += 1

                elif dev_type == 2 or dev_type == 3:
                    self.gens[dev_idx] = VRE(dev_idx, gen_idx, dev)
                    gen_idx += 1

                elif dev_type == 4:
                    self.storages[dev_idx] = Storage(dev_idx, su_idx, dev)
                    dev_idx += 1

                else:
                    raise ValueError(f'The DEV_TYPE attribute of case['
                                     f'{dev_idx, :} is not a valid type of '
                                     f'device.')
                dev_idx += 1

    def _build_admittance_matrix(self):
        """
        Build the nodal admittance matrix of the network (in p.u.).
        """

        Y_bus = np.zeros((self.N_bus, self.N_bus), dtype=np.complex)

        for br in self.branches:

            # Fill an off-diagonal elements of the admittance matrix Y_bus.
            Y_bus[br.f_bus, br.t_bus] = - (br.series / np.conjugate(br.tap))
            Y_bus[br.t_bus, br.f_bus] = - (br.series / br.tap)

            # Increment diagonal element of the admittance matrix Y_bus.
            Y_bus[br.f_bus, br.f_bus] += (br.series + br.shunt) \
                                         / (np.absolute(br.tap) ** 2)
            Y_bus[br.t_bus, br.t_bus] += br.series + br.shunt

        return Y_bus

    def _compute_bus_bounds(self):
        """
        Compute the range of (P, Q) possible injections at each bus.
        """

        P_min = [0.] * self.N_bus
        P_max = [0.] * self.N_bus
        Q_min = [0.] * self.N_bus
        Q_max = [0.] * self.N_bus

        # Iterate over all devices connected to the power grid.
        for dev in [self.slack_dev] + list(self.gens.values()) \
                    + list(self.loads.values()) + list(self.storages.values()):
            P_min[dev.bus_id] += dev.p_min
            P_max[dev.bus_id] += dev.p_max
            Q_min[dev.bus_id] += dev.q_min
            Q_max[dev.bus_id] += dev.q_max

        # Update each bus with its operation range.
        for idx, bus in enumerate(self.buses):
            bus.p_min = P_min[idx]
            bus.p_max = P_max[idx]
            bus.q_min = Q_min[idx]
            bus.q_max = Q_max[idx]

    def reset(self, init_soc=None):
        """
        Reset the simulator.

        Parameters
        ----------
        init_soc : list of float, optional
            The initial state of charge of each storage unit.
        """

        if init_soc is None:
            init_soc = [None] * self.N_storage

        # Reset the initial state of charge of each storage unit.
        for su in self.storages.values():
            if init_soc[su.type_id] is None:
                soc = self.rng.random() * (su.soc_max - su.soc_min) + su.soc_min
            else:
                soc = init_soc[su.type_id]
            su.soc = soc

    def get_network_specs(self):
        """
        Summarize the characteristics of the distribution network.

        Returns
        -------
        specs : dict of {str: list}
            A description of the operating range of the network. These values
            can be used to define an observation space.
        """

        P_min_bus = []
        P_max_bus = []
        Q_min_bus = []
        Q_max_bus = []
        V_min_bus = []
        V_max_bus = []

        P_min_dev = [0] * self.N_device
        P_max_dev = [0] * self.N_device
        Q_min_dev = [0] * self.N_device
        Q_max_dev = [0] * self.N_device
        dev_type = [0] * self.N_device

        soc_min = [0] * self.N_storage
        soc_max = [0] * self.N_storage

        I_max = []

        P_min_dev[0] = self.slack_dev.p_min
        P_max_dev[0] = self.slack_dev.p_max
        Q_min_dev[0] = self.slack_dev.q_min
        Q_max_dev[0] = self.slack_dev.q_max

        for dev in list(self.loads.values()) + list(self.gens.values()) \
                   + list(self.storages.values()):
            P_min_dev[dev.dev_id] = dev.p_min
            P_max_dev[dev.dev_id] = dev.p_max
            Q_min_dev[dev.dev_id] = dev.q_min
            Q_max_dev[dev.dev_id] = dev.q_max
            dev_type[dev.dev_id] = dev.type

        for su in self.storages.values():
            soc_min[su.type_id] = su.soc_min
            soc_max[su.type_id] = su.soc_max

        for branch in self.branches:
            I_max.append(branch.i_max)

        for bus in self.buses:
            P_min_bus.append(bus.p_min)
            P_max_bus.append(bus.p_max)
            Q_min_bus.append(bus.q_min)
            Q_max_bus.append(bus.q_max)
            V_min_bus.append(bus.v_min)
            V_max_bus.append(bus.v_max)

        specs = {'PMIN_BUS': P_min_bus, 'PMAX_BUS': P_max_bus,
                 'QMIN_BUS': Q_min_bus, 'QMAX_BUS': Q_max_bus,
                 'VMIN_BUS': V_min_bus, 'VMAX_BUS': V_max_bus,
                 'PMIN_DEV': P_min_dev, 'PMAX_DEV': P_max_dev,
                 'QMIN_DEV': Q_min_dev, 'QMAX_DEV': Q_max_dev,
                 'DEV_TYPE': dev_type, 'IMAX_BR': I_max,
                 'SOC_MIN': soc_min, 'SOC_MAX': soc_max}
        return specs

    def get_action_space(self):
        """
        Return the range of each possible control action.

        This function returns the lower and upper bound of each action that
        can be taken by the DSO.

        Returns
        -------
        P_curt_bounds : 2D numpy.ndarray
            The range of real power injection of VRE devices. For example,
            P_curt_bounds[i, :] = [p_max, p_min] of the i^th VRE generator (MW).
        alpha_bounds : 2D numpy.ndarray
            The operating range of each storage unit. For example,
            alpha_bounds[i, :] = [alpha_max, alpha_min] of each storage unit (MW).
        q_storage_bounds : 2D numpy.ndarray
            The range of reactive power injection of VRE devices. For example,
            q_storage_bounds[i, :] = [q_max, q_min] of each storage unit (MVAr).

        Notes
        -----
        The bounds returned by this function are loose, i.e. some parts of
        those spaces might be physically impossible to achieve due to other
        operating constraints. This is just an indication of the range of action
        available to the DSO.
        """

        P_curt_bounds = []
        for _, gen in sorted(self.gens.items()):
            if gen.type >= 2.:
                P_curt_bounds.append([gen.p_max, gen.p_min])

        alpha_bounds = []
        q_storage_bounds = []
        for _, su in sorted(self.storages.items()):
            alpha_bounds.append([su.p_max, su.p_min])
            q_storage_bounds.append([su.q_max, su.q_min])

        return np.array(P_curt_bounds), np.array(alpha_bounds), \
               np.array(q_storage_bounds)

    def transition(self, P_load, P_potential, P_curt_limit, desired_alpha,
                   Q_storage):
        """
        Simulate a transition of the system from time t to time (t+1).

        This function simulates a transition of the system after actions were
        taken by the DSO. The results of these decisions then affect the new
        state of the system, and the associated reward is returned.

        Parameters
        ----------
        P_load : array_like
            Real power injection from each load device (MW).
        P_potential : array_like
            Real power potential injection from each distributed generator (MW).
        P_curt_limit : array_like
            VRE curtailment instructions, excluding the slack bus (MW)
        desired_alpha : array_like
            Desired charging rate for each storage unit (MW).
        Q_storage : array_like
            Desired Q-setpoint of each storage unit (MVAr).

        Returns
        -------
        state : dict of {str : array_like}
            The current state of the system.
        reward : float
            The reward associated with the transition.
        e_loss : float
            The total energy loss.
        penalty : float
            The total penalty due to violation of operating constraints.
        """

        ### Manage passive loads. ###
        # 1. Compute the (P, Q) injection point of each load.
        for load in self.loads.values():
            load.compute_pq(P_load[load.type_id])

        ### Manage distributed generators. ###
        # 1. Curtail potential production.
        P_curt = np.minimum(P_potential, np.maximum(0, P_curt_limit))

        # 2. Compute the (P, Q) injection point of each generator
        # (except slack bus).
        for gen in self.gens.values():
            gen.compute_pq(P_curt[gen.type_id])

        # Initialize (P, Q) injection point of slack device to 0.
        self.slack_dev.p = 0.
        self.slack_dev.q = 0.

        ### Manage storage units. ###
        SOC = self._manage_storage(desired_alpha, Q_storage)

        ### Compute electrical quantities of interest. ###
        # 1. Compute total (P, Q) injection at each bus.
        self._get_bus_total_injections()

        # 2. Solve PFEs and compute nodal V (p.u.), P (MW), Q (MVAr) vectors.
        P_bus, Q_bus, V_bus = self._solve_pfes()

        # 3. Compute I in each branch (p.u.)
        I_br = self._get_branch_currents()
        I_br_magn = [np.absolute(i) for i in I_br]

        # 4. Compute P (MW) and Q (MVar) power flows in each branch.
        P_br, Q_br = self._compute_branch_PQ()

        # 5. Get (P, Q) injection of each device.
        P_dev = [self.slack_dev.p] + [0.] * (self.N_device - 1)
        Q_dev = [self.slack_dev.q] + [0.] * (self.N_device - 1)
        for dev in list(self.gens.values()) + list(self.loads.values()) + \
                   list(self.storages.values()):
            P_dev[dev.dev_id] = dev.p
            Q_dev[dev.dev_id] = dev.q

        ### Store all state variables in a dictionary. ###
        self.state = {'P_BUS': P_bus, 'Q_BUS': Q_bus, 'V_BUS': V_bus,
                      'P_DEV': P_dev, 'Q_DEV': Q_dev, 'SOC': SOC,
                      'P_BR': P_br, 'Q_BR': Q_br, 'IMAGN_BR': I_br_magn}

        ### Compute the total reward associated with the transition. ###
        reward, e_loss, penalty = self._get_reward(P_bus, P_potential, P_curt)

        return self.state, reward, e_loss, penalty

    def _manage_storage(self, desired_alpha, Q_storage_setpoints):
        """
        Manage all storage units during a transition.

        Parameters
        ----------
        desired_alpha : array_like
            The desired charging rate of each storage unit (MW).
        Q_storage_setpoints : array_like
            The desired Q-setpoint of each storage unit (MVAr).

        Returns
        -------
        SOC : list
            The state of charge of each storage unit (MWh).
        """

        SOC = [0.] * self.N_storage
        for su in self.storages.values():
            su.manage(desired_alpha[su.type_id], self.time_factor,
                      Q_storage_setpoints[su.type_id])
            SOC[su.type_id] = su.soc

        return SOC

    def _get_bus_total_injections(self):
        """
        Compute the total (P, Q) injection point at each bus.
        """

        P_bus = [0] * self.N_bus
        Q_bus = [0] * self.N_bus

        devs = [self.slack_dev] + list(self.gens.values()) \
               + list(self.loads.values()) + list(self.storages.values())
        for dev in devs:
            P_bus[dev.bus_id] += dev.p
            Q_bus[dev.bus_id] += dev.q

        for i, bus in enumerate(self.buses):
            bus.p = P_bus[i]
            bus.q = Q_bus[i]

    def _solve_pfes(self):
        """
        Solve the power flow equations and return V, P, Q for each bus.

        Returns
        -------
        P : numpy.ndarray
            The nodal real power nodal injection vector.
        Q : numpy.ndarray
            The nodal reactive power injection vector.
        V : numpy.ndarray
            The nodal complex voltage vector.

        Raises
        ------
        ValueError
            Raised if no solution is found.
        """

        P_bus, Q_bus = [], []
        for bus in self.buses[1:]:   # skip slack bus.
            P_bus.append(bus.p)
            Q_bus.append(bus.q)

        # Initialize complex V as v_ij = 1 exp(j 0).
        init_v = np.array([self.buses[0].v_slack] + [1.] * (self.N_bus - 1)
                          + [0.] * self.N_bus)

        # Transform (P, Q) injections into p.u. values.
        P_bus_pu = np.array(P_bus) / self.baseMVA
        Q_bus_pu = np.array(Q_bus) / self.baseMVA

        # Solve the power flow equations of the network.
        sol = optimize.root(self._power_flow_eqs, init_v,
                          args=(self.Y_bus, P_bus_pu, Q_bus_pu),
                          method='lm', options={'xtol': 1.0e-4})
        if not sol.success:
            raise ValueError('No solution to the PFEs: ', sol.message)
        x = sol.x

        # Re-create the complex bus voltage vector.
        V = x[0:self.N_bus] + 1.j * x[self.N_bus:]

        # Compute the complex power injection at the slack bus.
        s_slack = (V * np.conjugate(np.dot(self.Y_bus, V)))[0]

        # Retrieve the real and reactive power injections at the slack bus (=
        # slack device).
        self.slack_dev.p = np.real(s_slack) * self.baseMVA
        self.slack_dev.q = np.imag(s_slack) * self.baseMVA

        # Form the nodal P and Q vectors and convert from p.u. to MW and MVAr.
        P = np.hstack((self.slack_dev.p, P_bus_pu * self.baseMVA))
        Q = np.hstack((self.slack_dev.q, Q_bus_pu * self.baseMVA))

        for idx, bus in enumerate(self.buses):
            bus.v = V[idx]
            bus.p = P[idx]
            bus.q = Q[idx]

        return np.array(P), np.array(Q), np.array(V)

    def _power_flow_eqs(self, v, Y, P, Q):
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
            Fixed real power nodal injections, excluding the slack bus (p.u.).
        Q : 1D numpy.ndarray
            Fixed reactive power nodal injections, excluding the slack bus (p.u.).

        Returns
        -------
        2D numpy.ndarray
            A vector of 4 expressions to find the roots of.
        """

        # Re-build complex nodal voltage array.
        V = v[:self.N_bus] + 1.j * v[self.N_bus:]

        # Create complex equations as a matrix product.
        complex_rhs = V * np.conjugate(np.dot(Y, V))

        # Equations involving real variables and real power injections.
        real_p = np.real(complex_rhs[1:]) - P  # skip slack bus.

        # Equations involving real variables and reactive power injections.
        real_q = np.imag(complex_rhs[1:]) - Q  # skip slack bus.

        # Equation fixing the voltage magnitude at the slack bus.
        slack_magn = [np.absolute(V[0]) - self.buses[0].v_slack]

        # Equation fixing the voltage phase angle to be 0.
        slack_phase = [np.angle(V[0])]

        # Stack equations made of only real variables on top of each other.
        real_equations = np.hstack((real_p, real_q, slack_magn, slack_phase))

        return real_equations

    def _get_branch_currents(self):
        """
        Compute the complex current on each transmission line (in p.u.).

        Since branch currents are not symmetrical, the current injection at each
        end of the branch is computed, and the one with the highest magnitude
        is kept, i.e. |I_ij| = max(|I_ij|, |I_ji|).

        Returns
        -------
        numpy.ndarray
            The complex current in each transmission line (p.u.).
        """

        I_br = []
        for branch in self.branches:
            i_ij = self._get_current_from_voltage(branch, direction=True)
            i_ji = self._get_current_from_voltage(branch, direction=False)

            if np.abs(i_ij) >= np.abs(i_ji):
                branch.i = i_ij
            else:
                branch.i = - i_ji

            I_br.append(branch.i)

        return np.array(I_br)

    def _get_current_from_voltage(self, branch, direction=True):
        """
        Compute the complex current injection on a transmission line.

        Parameters
        ----------
        branch : `TransmissionLine`
            The transmission line in which to compute the current injection.
        direction : bool, optional
            True to compute the current injected into the branch from the
            `branch.f_bus` node; False to compute the injection from the
            `branch.t_bus` node.

        Returns
        -------
        complex
            The complex current injection into the transmission line (p.u.).
        """
        v_f = self.buses[branch.f_bus].v
        v_t = self.buses[branch.t_bus].v

        if direction:
            i_1 = (1. / (np.absolute(branch.tap) ** 2)) \
                  * (branch.series + branch.shunt) * v_f
            i_2 = - (1. / np.conjugate(branch.tap)) * branch.series * v_t

        else:
            i_1 = (branch.series + branch.shunt) * v_t
            i_2 = - (1. / branch.tap) * branch.series * v_f

        return i_1 + i_2

    def _compute_branch_PQ(self):
        """
        Compute the real and reactive power flows in each transmission line.

        Returns
        -------
        P_br : list of float
            The real power flow in each branch (MW).
        Q_br : list of float
            The reactive power flow in each branch (MVAr).
        """

        P_br, Q_br = [], []
        for branch in self.branches:
            s = self.buses[branch.f_bus].v * np.conj(branch.i) * self.baseMVA
            branch.p = np.real(s)
            branch.q = np.imag(s)

            P_br.append(branch.p)
            Q_br.append(branch.q)

        return P_br, Q_br

    def _get_reward(self, P_bus, P_potential, P_curt):
        """
        Return the total reward associated with the current state of the system.

        The reward is computed as a negative sum of transmission
        losses, curtailment losses, (dis)charging losses, and operational
        constraints violation costs.

        Parameters
        ----------
        P_bus : array_like
            The nodal real power injection vector (MW).
        P_potential : array_like
            The potential real power generation of each VRE (MW).
        P_curt : array_like
            The actual generation of each VRE after curtailment (MW).

        Returns
        -------
        reward : float
            The total reward associated with the transition to a new system
            state.
        e_loss : float
            The total energy loss.
        penalty : float
            The total penalty due to violation of operating constraints.
        """

        # Compute the total energy loss over the network. This includes
        # transmission losses, as well as energy sent to storage units.
        energy_loss = np.sum(P_bus) * self.time_factor

        # Compute energy loss due to curtailment. This is the difference
        # between the potential energy production, and the actual production.
        if P_curt.size:
            curt_loss = np.sum(np.max(P_potential - P_curt)) * self.time_factor
        else:
            curt_loss = 0.

        # Get the penalty associated with violating operating constraints.
        penalty = self._get_penalty()

        # Return reward as a negative cost.
        reward = - (energy_loss + curt_loss + penalty)
        return reward, energy_loss + curt_loss, penalty

    def _get_penalty(self):
        """
        Compute the penalty associated with operation constraints violation.

        Returns
        -------
        penalty : float
            The penalty associated with operation constraints violation.
        """

        # Compute the total voltage constraints violation (p.u.).
        v_penalty = 0.
        for bus in self.buses:
            v_magn = np.absolute(bus.v)
            v_penalty += np.maximum(0, v_magn - bus.v_max) \
                         + np.maximum(0, bus.v_min - v_magn)

        # Compute the total current constraints violation (p.u.).
        i_penalty = 0.
        for branch in self.branches:
            i_magn = np.absolute(branch.i)
            i_penalty += np.maximum(0, i_magn - branch.i_max)

        penalty = self.lamb * (v_penalty + i_penalty)
        return penalty
