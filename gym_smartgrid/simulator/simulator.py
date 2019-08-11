import numpy as np
import scipy.optimize as optimize

from gym_smartgrid.simulator.components import Bus, TransmissionLine, Load, \
    PowerPlant, \
    VRE, Storage
from gym_smartgrid.constants import DEV_H, BRANCH_H


class Simulator(object):
    """
    This class simulates an AC distribution system.
    """

    def __init__(self, case, delta_t=15., lamb=1e3, rng=None):
        """
        Initialize the state of the distribution network and all variables.

        :param case: a case object containing parameters of the grid.
        :param rng: a random seed.
        """

        # Check the correctness of the case file.
        #utils.check_casefile(case)

        self.delta_t = delta_t / 60.
        self.lamb = lamb

        # Initialize random generator.
        self.rng = np.random.RandomState() if rng is None else rng

        # Load the test case.
        self.baseMVA = case['baseMVA']

        self.buses = []
        for bus_id, bus in enumerate(case['bus']):
            if not bus_id:
                self.buses.append(Bus(bus, is_slack=True))
            else:
                self.buses.append(Bus(bus))

        self.branches = []
        for br in case['branch']:
            if br[BRANCH_H['BR_STATUS']]:
                self.branches.append(TransmissionLine(br))

        self.loads, self.gens, self.storages = {}, {}, {}
        dev_idx, load_idx, gen_idx, su_idx = 0, 0, 0, 0
        for dev in case['device']:
            if dev[DEV_H['DEV_STATUS']]:
                dev_type = dev[DEV_H['DEV_TYPE']]

                if dev_type == -1:
                    self.loads[dev_idx] = Load(dev_idx, load_idx, dev)
                    load_idx += 1

                elif dev_type == 0:
                    self.slack_dev = PowerPlant(dev_idx, gen_idx, dev, True)
                    gen_idx += 1

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

        # Number of elements in all sets.
        self.N_bus = len(self.buses)
        self.N_branch = len(self.branches)
        self.N_load = len(self.loads)
        self.N_gen = len(self.gens) + 1
        self.N_storage = len(self.storages)
        self.N_device = self.N_gen + self.N_load + self.N_storage

        # Build the nodal admittance matrix.
        self._build_admittance_matrix()

        self._compute_bus_bounds()
        self.network_specs = self.compute_network_specs()

    def _build_admittance_matrix(self):
        """
        Build the nodal admittance matrix of the network (in p.u.).

        This function builds the nodal admittance matrix of the network,
        based on specifications given in the input case file (in p.u.).
        """

        # Initialize an N-by-N empty complex admittance matrix.
        Y_bus = np.zeros((self.N_bus, self.N_bus), dtype=np.complex)

        # Initialize an N_branch-by-N_branch dict to store tap ratios,
        # series and shunt admittances of branches.
        taps = {}
        shunts = {}
        series = {}

        for branch in self.branches:
            # Compute the branch series admittance as y_s = 1 / (r + jx).
            y_series = 1. / (branch.r + 1.j * branch.x)

            # Compute the branch shunt admittance y_m = jb / 2.
            y_shunt = 1.j * branch.b / 2.

            # Create complex tap ratio of generator as: tap = a exp(j shift).
            shift = branch.shift * np.pi / 180.
            tap = branch.tap * np.exp(1.j * shift)

            # Fill an off-diagonal elements of the admittance matrix Y_bus.
            Y_bus[branch.f_bus, branch.t_bus] = - np.conjugate(
                tap) * y_series
            Y_bus[branch.t_bus, branch.f_bus] = - tap * y_series

            # Increment diagonal element of the admittance matrix Y_bus.
            Y_bus[branch.f_bus, branch.f_bus] += (y_series + y_shunt) \
                                                 * np.absolute(tap) ** 2
            Y_bus[branch.t_bus, branch.t_bus] += y_series + y_shunt

            # Store tap ratio, series admittance and shunt admittance.
            taps[(branch.f_bus, branch.t_bus)] = tap
            taps[(branch.t_bus, branch.f_bus)] = 1.
            shunts[(branch.f_bus, branch.t_bus)] = y_shunt
            shunts[(branch.t_bus, branch.f_bus)] = y_shunt
            series[(branch.f_bus, branch.t_bus)] = y_series
            series[(branch.t_bus, branch.f_bus)] = y_series

        self.Y_bus = Y_bus
        self.taps = taps
        self.shunts = shunts
        self.series = series

    def _compute_bus_bounds(self):
        P_min = [0.] * self.N_bus
        P_max = [0.] * self.N_bus
        Q_min = [0.] * self.N_bus
        Q_max = [0.] * self.N_bus

        for dev in [self.slack_dev] + list(self.gens.values()) \
                    + list(self.loads.values()) + list(self.storages.values()):
            P_min[dev.bus_id] += dev.p_min
            P_max[dev.bus_id] += dev.p_max
            Q_min[dev.bus_id] += dev.q_min
            Q_max[dev.bus_id] += dev.q_max

        for idx, bus in enumerate(self.buses):
            bus.p_min = P_min[idx]
            bus.p_max = P_max[idx]
            bus.q_min = Q_min[idx]
            bus.q_max = Q_max[idx]

    def reset(self):
        """ Reset the simulator. """
        self.state = None
        for su in self.storages.values():
            su.soc = su.soc_max / 2.

    def compute_network_specs(self):
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
        Return the upper and lower bound on each possible control action.

        This function returns the lower and upper bound of each action that
        can be taken by the DSO. More specifically, it returns 3 ndarray,
        each containing the upper and lower bounds of a type of action.

        For instance, P_curt_bounds[0, i] returns the maximum real power
        injection of generator i (skipping slack bus), and P_curt_bounds[1,
        i] its minimum injection.

        Note that the bounds returned by this
        function are loose, i.e. some parts of those spaces might be
        physically impossible to achieve due to other operating constraints.
        This is just an indication of the range of action available to the
        DSO.

        :return P_curt_bounds: bounds on the P generation of each generator
        device (ignoring slack device).
        :return alpha_bounds: bounds on the rate of charge of each storage unit.
        :return q_storage_bounds: bounds on the desired Q setpoint of each
        storage unit.
        """

        # Get bounds on the generation of each distributed generator (except
        # slack bus).
        P_curt_bounds = []
        for _, gen in sorted(self.gens.items()):
            if gen.type >= 2.:
                P_curt_bounds.append([gen.p_max, gen.p_min])

        # Get bounds on the charging rate and on the Q setpoint of each
        # storage unit.
        alpha_bounds = []
        q_storage_bounds = []
        for _, su in sorted(self.storages.items()):
            alpha_bounds.append([su.p_max, su.p_min])
            q_storage_bounds.append([su.q_max, su.q_min])

        return np.array(P_curt_bounds), np.array(alpha_bounds), \
               np.array(q_storage_bounds)

    def transition(self, P_load, P_potential, P_curt_limit, desired_alpha,
                   Q_storage_setpoints):
        """
        Simulates a transition of the system from time t to time (t+1).

        This function simulates a transition of the system after actions were
        taken by the DSO, during the previous time step. The results of these
        decisions then affect the new state of the system, and the associated
        reward is returned.

        :param P_load: N_load vector of real power injection from load devices.
        :param P_potential: (N_gen-1) vector of real power potential
        injections from distributed generators.
        :param P_curt_limit: (N_gen-1) vector of real power injection
        curtailment instructions, ignoring the slack bus (MW).
        :param desired_alpha: N_storage vector of desired charging rate for
        each storage unit (MW).
        :param Q_storage_setpoints: N_storage vector of desired Q-setpoint for
        storage units (MVAr).
        :return: the total reward associated with the transition.
        """

        ### Manage passive loads. ###
        # 1. Compute the reactive power injection at each load.
        for load in self.loads.values():
            load.compute_pq(P_load[load.type_id])

        ### Manage distributed generators. ###
        # 1. Curtail potential production (except slack bus).
        P_curt = np.minimum(P_potential, np.maximum(0, P_curt_limit))

        # 2. Get (P, Q) feasible injection points (except slack bus).
        for gen in self.gens.values():
            gen.compute_pq(P_curt[gen.type_id - 1])

        ### Manage storage units. ###
        SOC = self._manage_storage(desired_alpha, Q_storage_setpoints)

        ### Compute electrical quantities of interest. ###
        # 1. Compute total P and Q injection at each bus.
        self._get_bus_total_injections()

        # 2. Solve PFEs and store nodal V (p.u.), P (MW), Q (MVAr) vectors.
        P_bus, Q_bus, V_bus = self._solve_pfes()

        # 3. Compute I in each branch from V (p.u.)
        I_br = self._get_branch_currents()

        # 4. Compute P (MW), Q (MVar) power flows in each branch.
        P_br, Q_br = self._compute_branch_PQ()

        # 5. Get (P, Q) injection of each device.
        P_dev = [self.slack_dev.p] + [0.] * (self.N_device - 1)
        Q_dev = [self.slack_dev.q] + [0.] * (self.N_device - 1)
        for dev in list(self.gens.values()) + list(self.loads.values()) + \
                   list(self.storages.values()):
            P_dev[dev.dev_id] = dev.p
            Q_dev[dev.dev_id] = dev.q

        # 6. Store all state variables in a dictionary.
        self.state = {'P_BUS': P_bus, 'Q_BUS': Q_bus, 'V_BUS': V_bus,
                      'P_DEV': P_dev, 'Q_DEV': Q_dev, 'SOC': SOC,
                      'P_BR': P_br, 'Q_BR': Q_br, 'I_BR': I_br}

        ### Return the total reward associated with the transition. ###
        reward = self._get_reward(P_bus, P_potential, P_curt, I_br, V_bus)

        return reward

    def _manage_storage(self, desired_alpha, Q_storage_setpoints):
        SOC = [0.] * self.N_storage
        for su in self.storages.values():
            su.manage(desired_alpha[su.type_id], self.delta_t,
                      Q_storage_setpoints[su.type_id])
            SOC[su.type_id] = su.soc

        return SOC

    def _get_bus_total_injections(self):
        """
        """

        P_bus = [0] * self.N_bus
        Q_bus = [0] * self.N_bus
        devs = list(self.gens.values()) + list(self.loads.values()) + \
                   list(self.storages.values())
        for dev in devs:
            P_bus[dev.bus_id] += dev.p
            Q_bus[dev.bus_id] += dev.q

        for i, bus in enumerate(self.buses):
            bus.p = P_bus[i]
            bus.q = Q_bus[i]

        return

    def _solve_pfes(self):
        """
        Solve the power flow equations and return V, P, Q for each bus.

        This function solves the power flow equations of the network. If no
        solution is found, a ValueError is raised and a message displayed. The
        real and reactive power injections at the slack bus are then retrieved
        from the solution V, and the nodal vectors V, P, Q are reconstructed.

        :return: the N vectors of nodal V (p.u.), P (MW), and Q (MVAr).
        """

        P_bus, Q_bus = [], []
        for bus in self.buses[1:]:
            P_bus.append(bus.p)
            Q_bus.append(bus.q)

        # Initialize V to represent v_ij = 1 exp(j 0).
        init_v = np.array([self.buses[0].v_slack] + [1.] * (self.N_bus - 1)
                          + [0.] * self.N_bus)

        # Transform P, Q injections into p.u. values.
        P_bus_pu = np.array(P_bus) / self.baseMVA # skip slack bus.
        Q_bus_pu = np.array(Q_bus) / self.baseMVA # skip slack bus.

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
        # slack device, since there is only 1 device at the slack bus).
        self.slack_dev.p = np.real(s_slack)
        self.slack_dev.q = np.imag(s_slack)

        # Form the nodal P and Q vectors and convert from p.u. to MW and MVAr.
        P = np.hstack((self.slack_dev.p, P_bus_pu)) * self.baseMVA
        Q = np.hstack((self.slack_dev.q, Q_bus_pu)) * self.baseMVA

        for idx, bus in enumerate(self.buses):
            bus.v = V[idx]
            bus.p = P[idx]
            bus.q = Q[idx]

        return np.array(P), np.array(Q), np.array(V)

    def _power_flow_eqs(self, v, Y, P, Q):
        """
        Return the power flow equations to be solved.

        This is a vector function, returning a vector of expressions which
        must be equal to 0 (i.e. find the roots) to solve the Power Flow
        Equations of the network. Everything should be expressed in p.u.

        :param v: ndarray of size (2N,), where elements 0 to N-1 represent the
        real part of the initial guess for nodal voltage, and the N to 2N-1
        elements their corresponding imaginary parts (p.u.).
        :param Y: the network bus admittance matrix (p.u.).
        :param P: ndarray of size N-1 of fixed real power nodal injections,
        excluding the slack bus (p.u.).
        :param Q: ndarray of size N-1 of fixed reactive power nodal
        injections, excluding the slack bus (p.u.).
        :return: a vector of 4 expressions to find the roots of.
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
        Return the current magnitude on each transmission line (in p.u.).

        This function returns the magnitude of the current I_ij, for each
        transmission line (i, j). Since branch
        currents are not symmetrical, the current magnitude in branch (i,
        j) is taken to be max(abs(I_ij, I_ji)).

        :return: a list of branch current magnitudes (p.u.).
        """
        I_br = []
        for branch in self.branches:
            f_bus, t_bus = branch.f_bus, branch.t_bus

            f_v = self.buses[f_bus].v
            t_v = self.buses[t_bus].v

            i_ij = self._get_current_from_voltage(f_bus, t_bus, f_v, t_v)
            i_ji = self._get_current_from_voltage(t_bus, f_bus, t_v, f_v)

            # Store the current with the biggest current magnitude.
            if np.abs(i_ij) >= np.abs(i_ji):
                branch.i = i_ij
            else:
                branch.i = - i_ji

            I_br.append(branch.i)

        return np.array(I_br)

    def _get_current_from_voltage(self, i, j, v_i, v_j):
        """
        Return the current sent on transmission line (i, j) at node i.

        This function computes the line complex current from bus i to bus j,
        based on voltage values. This is the current value as it leaves node
        i. Note that I_ij is different from I_ji.

        :param i: the sending end of the line.
        :param j: the receiving end of the line.
        :return: the complex current on the desired transmission line (p.u.).
        """

        # Get the characteristics of the transmission line and transformer.
        tap = self.taps[(i, j)]
        y_s = self.series[(i, j)]
        y_shunt = self.shunts[(i, j)]

        # Compute the complex current in the branch, as seen from node i (p.u.).
        current = np.absolute(tap) ** 2 * (y_s + y_shunt) * v_i - \
                  np.conjugate(tap) * y_s * v_j

        return current

    def _compute_branch_PQ(self):
        """ Return P (MW), Q (MVar) flow in each branch. """

        P_br, Q_br = [], []
        for branch in self.branches:
            s = self.buses[branch.f_bus].v * np.conj(branch.i)
            branch.p = np.real(s)
            branch.q = np.imag(s)

            P_br.append(branch.p)
            Q_br.append(branch.q)

        return P_br, Q_br

    def _get_reward(self, P_bus, P_potential, P_curt, I_br, V_bus):
        """
        Return the total reward associated with the current state of the system.

        The reward is computed as a negative sum of transmission
        losses, curtailment losses, (dis)charging losses, and operational
        constraints violation costs.

        :param P_potential: the vector of potential nodal real generation (MW).
        :return: the total reward associated with the current system state.
        """

        # Compute the total energy loss over the network. This includes
        # transmission losses, as well as energy sent to storage units.
        energy_loss = np.sum(P_bus) * self.delta_t

        # Compute energy loss due to curtailment. This is the difference
        # between the potential energy production, and the actual production.
        if P_curt.size:
            curt_loss = np.sum(np.max(P_potential - P_curt)) * self.delta_t
        else:
            curt_loss = 0.

        # Get the penalty associated with violating operating constraints.
        penalty = self._get_penalty(V_bus, I_br)

        # Return reward as a negative cost.
        reward = - (energy_loss + curt_loss + penalty)
        return reward

    def _get_penalty(self, V_bus, I_br):
        """
        Return the penalty associated with operation constraints violation.

        This function returns a (big) penalty cost if the system violates
        operation constraints, that is voltage magnitude and line current
        constraints.

        :return: the penalty associated with operation constraints violation.
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
