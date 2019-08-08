import numpy as np
import warnings
import scipy.optimize as optimize


class Simulator(object):
    """
    This class simulates an AC distribution system.
    """

    # Create dictionary of {name, column_idx} to map electrical quantities to
    # column numbers in the case file.
    headers_bus = ['BUS_I', 'BUS_TYPE', 'PD', 'QD', 'GS', 'BS', 'BUS_AREA',
                   'VM', 'VA', 'BASE_KV', 'ZONE', 'VMAX', 'VMIN']
    BUS_H = dict(zip(headers_bus, range(len(headers_bus))))

    headers_gen = ['GEN_BUS', 'PG', 'QG', 'QMAX', 'QMIN', 'VG', 'MBASE',
                   'DEV_STATUS', 'PMAX', 'PMIN', 'PC1', 'PC2', 'QC1MIN',
                   'QC1MAX', 'QC2MIN', 'QC2MAX', 'VRE_TYPE']
    GEN_H = dict(zip(headers_gen, range(len(headers_gen))))

    headers_branch = ['F_BUS', 'T_BUS', 'BR_R', 'BR_X', 'BR_B', 'RATE_A',
                      'RATE_B', 'RATE_C', 'TAP', 'SHIFT', 'BR_STATUS',
                      'ANGMIN', 'ANGMAX']
    BRANCH_H = dict(zip(headers_branch, range(len(headers_branch))))

    headers_storage = ['BUS_I', 'SOC_MAX', 'EFF', 'PMAX', 'QMAX',
                       'PC1', 'PC2', 'QC1MIN', 'QC1MAX', 'QC2MIN',
                       'QC2MAX']
    STORAGE_H = dict(zip(headers_storage, range(len(headers_storage))))

    # Factor to use to get values pee hour (i.e. 0.25 = time step of 15 minutes).
    DELTA_T = 0.25

    # Constant used to make the penalty due to constraints violation big.
    LAMB = 1e3


    def __init__(self, case, rng=None):
        """
        Initialize the state of the distribution network and all variables.

        :param case: a case object containing parameters of the grid.
        :param rng: a random seed.
        """

        # Initialize random generator.
        self.rng = np.random.RandomState() if rng is None else rng

        # Load the test case, selecting only active branches.
        self.buses = case['bus']
        self.branches = case['branch'][case['branch']
                                       [:, Simulator.BRANCH_H['BR_STATUS']] ==
                                       1., :]
        self.gen = case['gen']
        self.storage = case['storage']
        self.baseMVA = case['baseMVA']

        # Check that the slack bus is correctly specified in the input case.
        self._init_check_slack_bus()

        # Number of elements in all sets.
        self.N_bus = self.buses.shape[0]
        self.N_branch = self.branches.shape[0]
        self.N_device = self.gen.shape[0] + self.storage.shape[0]
        self.N_storage = self.storage.shape[0]

        # Store pairs of buses representing transmission lines (0-indexed).
        lines = []
        for i in range(self.N_branch):
            f_bus =  self.branches[i, self.BRANCH_H['F_BUS']] - 1
            t_bus = self.branches[i, self.BRANCH_H['T_BUS']] - 1
            lines.append((int(f_bus), int(t_bus)))
        self.lines = lines

        # Create mapping dictionaries between device and bus indices.
        self._init_build_device_bus_mapping()

        # Create restriction on (P, Q) injection points of devices.
        self._init_build_pq_rules()

        # Build the nodal admittance matrix.
        self._init_build_admittance_matrix()

        # Get bounds on current magnitudes in all branches (in p.u.).
        self.Imax = []
        for branch in self.branches:
            if branch[Simulator.BRANCH_H['RATE_A']] > 0.:
                imax = branch[Simulator.BRANCH_H['RATE_A']] / self.baseMVA
            else:
                imax = np.inf
            self.Imax.append(imax)

        # Get bounds on voltage magnitude at all buses, except the slack bus (
        # in p.u.).
        self.Vmin = self.buses[:, Simulator.BUS_H['VMIN']]
        self.Vmax = self.buses[:, Simulator.BUS_H['VMAX']]

        # Get voltage magnitude set-point at slack bus (in p.u.).
        self.V_magn_slack = self.gen[0, Simulator.GEN_H['VG']]

        # Check that the set-point voltage magnitude at the slack bus
        # seems coherent.
        if (self.V_magn_slack < 0.5) or (self.V_magn_slack > 1.5):
            warnings.warn("Warning: voltage magnitude (" + str(self.V_magn_slack)
                          + ") at the slack bus does not seem coherent.")

        # Get the maximum SoC for each storage unit.
        self.max_soc = self.storage[:, Simulator.STORAGE_H['SOC_MAX']]

        # Initialize state of charge (SoC) of storage units.
        self.SoC = self.max_soc / 2.

        # Initialize variables.
        self.P_device, self.Q_device = None, None
        self.P_br, self.Q_br, self.I_br, self.I_br_magn = None, None, None, None

    def reset(self):
        """ Reset the simulator. """

        self.SoC = self.max_soc / 2.

    def get_vre_specs(self):
        """
        Return the maximum real power output of each load and VRE device.

        This function returns 2 dictionary of the type {dev_idx: Pmax},
        one for wind resources and one for solar resources. It also returns a
        list of [Pmax] containing the maximum real power output of each load.
        It is used to initialize the stochastic processes modelling all loads
        and VRE resources in the network.

        :return: 2 dicts and 1 list, containing Pmax for each load and vre
        device.
        """

        wind, solar, load = {}, {}, []
        for dev_idx in range(self.gen.shape[0]):
            if self.gen[dev_idx, self.GEN_H['VRE_TYPE']] == -1:
                load.append(self.gen[dev_idx, self.GEN_H['PMIN']])
            if self.gen[dev_idx, self.GEN_H['VRE_TYPE']] == 1.:
                wind[dev_idx] = self.gen[dev_idx, self.GEN_H['PMAX']]
            if self.gen[dev_idx, self.GEN_H['VRE_TYPE']] == 2.:
                solar[dev_idx] = self.gen[dev_idx, self.GEN_H['PMAX']]

        return wind, solar, load

    def get_network_specs(self):

        dev_type = list(self.gen[:, Simulator.GEN_H['VRE_TYPE']])

        P_min = [self.gen[0, self.GEN_H['PMIN']]]
        P_max = [self.gen[0, self.GEN_H['PMAX']]]
        for dev_idx in range(1, self.N_device):
            rule = self.pq_rules[dev_idx]
            P_min.append(rule.pmin)
            P_max.append(rule.pmax)

        soc_min = []
        soc_max = []
        for storage_idx in range(self.N_storage):
            soc_min.append(0)
            soc_max.append(self.max_soc[storage_idx])

        return dev_type, P_min, P_max, self.Imax, soc_min, soc_max

    def _init_build_device_bus_mapping(self):
        """
        Creates 2 dictionaries to easily mapping between devices and buses.

        This function creates two dictionaries:
            1. dev2bus: {device_idx: bus_idx},
            2. type2bus: {device_type: device_idx},
            3. dev2storage: {device_idx: storage_idx}
        where 'device_type' can take the following values: slack, generator,
        load, storage. 'storage_idx' refers to the index of the storage units
        in the case['storage'] table.

        Note: all values are 0-indexed. So, all bus number are shifted by 1
        from the original case file.
        """

        # Initialize an empty dictionary to create a mapping device -> bus.
        dev2bus = {}
        # Initialize empty dictionary to map device_idx -> storage_idx.
        dev2storage = {}
        # Initialize empty dictionary to map device_idx -> gen_idx.
        dev2gen = {}
        # Initialize empty dictionary to map device_idx -> load_idx.
        dev2load = {}

        # Add indices of generators and loads (0-indexing).
        idx_dev, idx_gen, idx_load = 0, 0, 0
        for i in range(self.gen.shape[0]):
            dev2bus[i] = int(self.gen[i, Simulator.GEN_H['GEN_BUS']]) - 1

            if self.gen[i, Simulator.GEN_H['PG']] > 0.:
                if i > 0: # skip slack bus.
                    dev2gen[i] = idx_gen
                idx_gen += 1
            elif self.gen[i, Simulator.GEN_H['PG']] < 0.:
                dev2load[i] = idx_load
                idx_load += 1
            idx_dev = i

        idx_dev += 1

        # Add indices of storage units.
        for i in range(idx_dev, idx_dev + self.N_storage):
            dev2bus[i] = int(self.storage[i - idx_dev, 0]) - 1
            dev2storage[i] = i - idx_dev

        self.dev2bus = dev2bus
        self.dev2gen = dev2gen
        self.dev2load = dev2load
        self.dev2storage = dev2storage

        # Initialize an empty dictionary to store lists of indices to easily
        # retrieve all generators / loads / storage units.
        type2dev = {'slack': [0]} # initialize with slack bus device.

        # Add distributed generators.
        type2dev['gen'] = np.nonzero(self.gen[:, Simulator.GEN_H['PG']] > 0.)[0]
        type2dev['load'] = np.nonzero(self.gen[:, Simulator.GEN_H['PG']] < 0.)[0]
        type2dev['storage'] = np.arange(idx_dev, idx_dev + self.N_storage)

        self.type2dev = type2dev

    def _init_build_pq_rules(self):
        """
        Get the region of power injection capabilities for a load or gen.

        This function creates a PowerCapabilities object, for each passive
        load and distributed generator, storing the information needed to
        define the corresponding region of feasible (P, Q) injections.
        Everything is expressed in in MW or MVAr.
        """
        self.pq_rules = {}

        # Store restrictions for slack bus.
        for dev_idx in self.type2dev['slack']:
            rule = PowerCapabilities('slack bus')
            rule.pmin = self.gen[dev_idx, Simulator.GEN_H['PMIN']]
            rule.pmax = self.gen[dev_idx, Simulator.GEN_H['PMAX']]
            self.pq_rules[dev_idx] = rule

        # Store restrictions for passive load power injections.
        for dev_idx in self.type2dev['load']:
            rule = PowerCapabilities('passive load')

            rule.pmin = self.gen[dev_idx, Simulator.GEN_H['PMIN']]
            rule.pmax = self.gen[dev_idx, Simulator.GEN_H['PMAX']]

            rule.qp_ratio = self.gen[dev_idx, Simulator.GEN_H['QG']] / \
                            self.gen[dev_idx, Simulator.GEN_H['PG']]
            self.pq_rules[dev_idx] = rule

        # Store restrictions for distributed generators.
        indices = (self.type2dev['gen'])
        for dev_idx in indices:
            rule = PowerCapabilities('distributed generator')

            rule.qp_ratio = self.gen[dev_idx, Simulator.GEN_H['QG']] / \
                            self.gen[dev_idx, Simulator.GEN_H['PG']]


            rule.pmin = self.gen[dev_idx, Simulator.GEN_H['PMIN']]
            rule.qmin = self.gen[dev_idx, Simulator.GEN_H['QMIN']]

            if self.gen[dev_idx, Simulator.GEN_H['PMAX']] > 0.:
                rule.pmax = self.gen[dev_idx, Simulator.GEN_H['PMAX']]

            if self.gen[dev_idx, Simulator.GEN_H['QMAX']] > 0.:
                rule.qmax = self.gen[dev_idx, Simulator.GEN_H['QMAX']]

            if self.gen[dev_idx, Simulator.GEN_H['PC1']] \
                != self.gen[dev_idx, Simulator.GEN_H['PC2']]:
                rule.lead_limit = {k: self.gen[dev_idx, Simulator.GEN_H[k]] for
                                   k in ['PC1', 'PC2', 'QC1MAX', 'QC2MAX']}
                rule.lag_limit = {k: self.gen[dev_idx, Simulator.GEN_H[k]] for
                                  k in ['PC1', 'PC2', 'QC1MIN', 'QC2MIN']}

            self.pq_rules[dev_idx] = rule

        # Store restrictions for storage units.
        for dev_idx in self.type2dev['storage']:
            rule = PowerCapabilities('storage unit')
            storage_idx = self.dev2storage[dev_idx]

            if self.storage[storage_idx, Simulator.STORAGE_H['PMAX']] > 0.:
                rule.pmax = self.storage[storage_idx,
                                         Simulator.STORAGE_H['PMAX']]
                rule.pmin = 0.

            if self.storage[storage_idx, Simulator.STORAGE_H['QMAX']] > 0.:
                rule.qmax = self.storage[storage_idx,
                                         Simulator.STORAGE_H['QMAX']]
                rule.qmin = - rule.qmax

            if self.storage[storage_idx, Simulator.STORAGE_H['PC1']] \
                != self.storage[storage_idx, Simulator.STORAGE_H['PC2']]:
                rule.lead_limit = {k: self.storage[storage_idx,
                                                   Simulator.STORAGE_H[k]]
                                   for k in ['PC1', 'PC2', 'QC1MAX', 'QC2MAX']}
                rule.lag_limit = {k: self.storage[storage_idx,
                                                  Simulator.STORAGE_H[k]]
                                  for k in ['PC1', 'PC2', 'QC1MIN', 'QC2MIN']}

            self.pq_rules[dev_idx] = rule

    def _init_check_slack_bus(self):
        """
        Check that the slack bus is properly defined in the input case file.

        This function checks the following:
            1. There is exactly 1 bus specified as the slack bus (TYPE = 3).
            2. The slack bus is the first bus in the case['bus'] input.
            3. That there is exactly 1 PV generator device: VG != 0.
            4. Check that the PV generator is the first device specified in
               case['gen'].
        """

        # Check if there is exactly 1 slack bus, as specified in the case file.
        if np.sum(self.buses[:, Simulator.BUS_H['BUS_TYPE']] == 3) != 1:
            raise ValueError('There should be exactly one slack bus, '
                             'as specified in the TYPE field of case["bus"].')

        # Check the correctness of the slack bus specifications and get its
        # fixed voltage magnitude.
        if self.buses[0, Simulator.BUS_H['BUS_TYPE']] == 3:

            # Check devices that are given as PV variables (fixed P and |V|).
            # There should be exactly one such device (the slack bus) and it
            # should be the first bus in the input file list. An error is
            # raised otherwise.

            # Check if there is exactly one PV generator.
            if np.sum(self.gen[:, Simulator.GEN_H['VG']] != 0.) != 1:
                raise ValueError('There should be exactly 1 PV generator '
                                 'connected to the slack (first) bus.')

            # Check if the PV generator is the first one in the list of
            # generators specified.
            if self.gen[0, Simulator.GEN_H['VG']] == 0.:
                raise ValueError('The first generator in the input file should '
                                 'be a PV generator, connected to the slack '
                                 'bus.')
            # Check if there is exactly 1 slack device specified in the
            # VRE_TYPE column and it is the first one.
            if np.sum(self.gen[:, Simulator.GEN_H['VRE_TYPE']] == 0.) != 1 \
                or self.gen[0, Simulator.GEN_H['VRE_TYPE']] != 0.:
                raise ValueError('The first device in the case.gen table '
                                 'should have VRE_TYPE == 0. to signify slack '
                                 'bus, and no other device should.')

        else:
            raise ValueError("The slack bus of the test case must be specified "
                             "as the first bus in the input file. case['bus']["
                             "0, 1] == 3 should be true.")

    def _init_build_admittance_matrix(self):
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

        # Build the complex nodal admittance matrix, iterating over branches.
        for branch_idx, branch in enumerate(self.branches):

            # Get sending and receiving buses of the branch (0 indexing).
            bus_f, bus_t = self.lines[branch_idx]

            # Compute the branch series admittance as y_s = 1 / (r + jx).
            y_series = 1. / (branch[Simulator.BRANCH_H['BR_R']]
                             + 1.j * branch[Simulator.BRANCH_H['BR_X']])

            # Compute the branch shunt admittance y_m = jb / 2.
            y_shunt = 1.j * branch[Simulator.BRANCH_H['BR_B']] / 2.

            # Get the tap ratio of the transformer.
            tap = branch[Simulator.BRANCH_H['TAP']]
            tap = tap if tap > 0. else 1.

            # Get the angle shift in radians.
            shift = branch[Simulator.BRANCH_H['SHIFT']] * np.pi / 180.

            # Create complex tap ratio of generator as: tap = a exp(j shift).
            tap = tap * np.exp(1.j * shift)

            # Fill an off-diagonal elements of the admittance matrix Y_bus.
            Y_bus[bus_f, bus_t] = - np.conjugate(tap) * y_series
            Y_bus[bus_t, bus_f] = - tap * y_series

            # Increment diagonal element of the admittance matrix Y_bus.
            Y_bus[bus_f, bus_f] += (y_series + y_shunt) * np.absolute(tap) ** 2
            Y_bus[bus_t, bus_t] += y_series + y_shunt

            # Store tap ratio, series admittance and shunt admittance.
            taps[(bus_f, bus_t)] = tap
            taps[(bus_t, bus_f)] = 1.
            shunts[(bus_f, bus_t)] = y_shunt
            shunts[(bus_t, bus_f)] = y_shunt
            series[(bus_f, bus_t)] = y_series
            series[(bus_t, bus_f)] = y_series

        self.Y_bus = Y_bus
        self.taps = taps
        self.shunts = shunts
        self.series = series

    def _get_reward(self, P_potential, P_curt):
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
        energy_loss = np.sum(self.P) * Simulator.DELTA_T

        # Compute energy loss due to curtailment. This is the difference
        # between the potential energy production, and the actual production.
        if P_curt.size > 0:
            curt_loss = np.sum(np.max(P_potential - P_curt)) * Simulator.DELTA_T
        else:
            curt_loss = 0.

        # Get the penalty associated with violating operating constraints.
        penalty = self._get_penalty()

        # Return reward as a negative cost.
        reward = - (energy_loss + curt_loss + penalty)
        return reward

    def _get_penalty(self):
        """
        Return the penalty associated with operation constraints violation.

        This function returns a (big) penalty cost if the system violates
        operation constraints, that is voltage magnitude and line current
        constraints.

        :return: the penalty associated with operation constraints violation.
        """

        # Compute the total voltage constraints violation (p.u.).
        V_magn = np.absolute(self.V)
        V_penalty = np.sum(np.maximum(0, V_magn - self.Vmax) \
                           + np.maximum(0, self.Vmin - V_magn))

        # Compute the total current constraints violation (p.u.).
        I_penalty = 0.
        for branch_idx in range(self.N_branch):

            # Get the current magnitude on the branch.
            i_magn = self.I_br_magn[branch_idx]

            # Get maximum branch current (p.u.).
            i_max = self.Imax[branch_idx]

            I_penalty += np.maximum(0, i_magn - i_max)

        penalty = Simulator.LAMB * (V_penalty + I_penalty)

        # Set the is_safe flag to indicate if operation constraints are violated.
        self.is_safe = False if penalty > 0. else True

        return penalty

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
        gen_indices = self.type2dev['gen']
        P_curt_bounds = np.zeros(shape=(2, len(gen_indices)))
        for gen_idx, dev_idx in enumerate(gen_indices):
            rule = self.pq_rules[dev_idx]
            P_curt_bounds[0, gen_idx] = rule.pmax
            P_curt_bounds[1, gen_idx] = rule.pmin

        # Get bounds on the charging rate and on the Q setpoint of each
        # storage unit.
        storage_indices = self.type2dev['storage']
        alpha_bounds = np.zeros(shape=(2, self.N_storage))
        q_storage_bounds = np.zeros(shape=(2, self.N_storage))
        for storage_idx, dev_idx in enumerate(storage_indices):
            rule = self.pq_rules[dev_idx]
            alpha_bounds[0, storage_idx] = rule.pmax
            alpha_bounds[1, storage_idx] = - rule.pmax
            q_storage_bounds[0, storage_idx] = rule.qmax
            q_storage_bounds[1, storage_idx] = rule.qmin

        return P_curt_bounds, alpha_bounds, q_storage_bounds

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
        Q_load = self._get_reactive_injection_load(P_load)

        ### Manage distributed generators. ###
        # 1. Curtail potential production (except slack bus).
        P_curt = np.minimum(P_potential, np.maximum(0, P_curt_limit))

        # 2. Get (P, Q) feasible injection points (except slack bus).
        P_gen, Q_gen = self._get_generator_injection_points(P_curt)

        ### Manage storage units. ###
        # 1. Get P, delta_soc for each storage unit.
        P_storage, delta_soc = self._manage_storage(desired_alpha)

        # 2. Get reactive power injection at storage units.
        Q_storage = self._get_reactive_injection_storage(P_storage,
                                                         Q_storage_setpoints)

        # 3. Update SoC of each storage unit.
        self.SoC = [soc + delta for soc, delta in zip(self.SoC, delta_soc)]

        ### Compute electrical quantities of interest. ###
        # 1. Compute total P and Q injection at each bus.
        Ps = {'generator': P_gen, 'load': P_load, 'storage': P_storage}
        Qs = {'generator': Q_gen, 'load': Q_load, 'storage': Q_storage}
        P_bus, Q_bus = self._get_bus_total_injections(Ps, Qs)

        # 2. Solve PFEs and store nodal V (p.u.), P (MW), Q (MVAr) vectors.
        self.V, self.P, self.Q = self._solve_pfes(P_bus, Q_bus)

        # 3. Build the vectors of P and Q injections from devices.
        self.P_device, self.Q_device = self._get_device_P(Ps, Qs)

        # 4. Compute I in each branch from V (p.u.)
        self.I_br, self.I_br_magn = self._get_branch_currents()

        # 5. Compute P (MW), Q (MVar) power flows in each branch.
        self.P_br, self.Q_br = self._compute_branch_PQ()

        ### Return the total reward associated with the transition. ###
        return self._get_reward(P_potential, P_curt)

    def _compute_branch_PQ(self):
        """ Return P (MW), Q (MVar) flow in each branch. """
        S_branch = []
        for line_idx, line in enumerate(self.lines):
            s_branch = self.V[line[0]] * np.conj(self.I_br[line_idx])
            S_branch.append(s_branch * self.baseMVA)

        P_branch = [np.real(s) for s in S_branch]
        Q_branch = [np.imag(s) for s in S_branch]

        return P_branch, Q_branch

    def _get_device_P(self, Ps, Qs):
        P_device = [0] * self.N_device
        Q_device = [0] * self.N_device

        # P and Q injection from single device at slack bus.
        P_device[0] = self.P[0]
        Q_device[0] = self.Q[0]

        for dev_idx in self.type2dev['gen']:
            gen_idx = self.dev2gen[dev_idx]
            P_device[dev_idx] = Ps['generator'][gen_idx]
            Q_device[dev_idx] = Qs['generator'][gen_idx]
        for dev_idx in self.type2dev['load']:
            load_idx = self.dev2load[dev_idx]
            P_device[dev_idx] = Ps['load'][load_idx]
            Q_device[dev_idx] = Qs['load'][load_idx]
        for dev_idx in self.type2dev['storage']:
            storage_idx = self.dev2storage[dev_idx]
            P_device[dev_idx] = Ps['storage'][storage_idx]
            Q_device[dev_idx] = Qs['storage'][storage_idx]

        return P_device, Q_device

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
        for branch_idx, branch in enumerate(self.branches):

            # Get sending and receiving bus indices.
            i, j = self.lines[branch_idx]

            # Compute the current I_ij and I_ji (not symmetrical).
            i_ij = self._get_current_from_voltage(i, j)
            i_ji = self._get_current_from_voltage(j, i)

            # Store the current with the biggest current magnitude.
            if np.abs(i_ij) >= np.abs(i_ji):
                I_br.append(i_ij)
            else:
                I_br.append(- i_ji)

        I_br_magn = [np.abs(i) for i in I_br]

        return I_br, I_br_magn

    def _get_bus_total_injections(self, Ps, Qs):
        """
        Return the total real and reactive power injection at each bus.

        :param Ps: a dict of lists of real power injection with keys {
        'generator', 'load', 'storage'}.
        :param Qs: a dict of lists of reactive power injection with keys {
        'generator', 'load', 'storage'}.
        :return: the total real (MW) and reactive (MVAr) power injection at each
        bus, except the slack bus.
        """

        P_total = np.zeros(shape=(self.N_bus)) # ignore slack bus.
        Q_total = np.zeros(shape=(self.N_bus))

        # Iterate over generator devices.
        gen_idx = 0
        for dev_idx in self.type2dev['gen']:
            bus_idx = self.dev2bus[dev_idx]
            P_total[bus_idx] += Ps['generator'][gen_idx]
            Q_total[bus_idx] += Qs['generator'][gen_idx]
            gen_idx += 1

        # Iterate over passive load devices.
        load_idx = 0
        for dev_idx in self.type2dev['load']:
            bus_idx = self.dev2bus[dev_idx]
            P_total[bus_idx] += Ps['load'][load_idx]
            Q_total[bus_idx] += Qs['load'][load_idx]
            load_idx += 1

        # Iterate over storage units.
        for dev_idx in self.type2dev['storage']:
            storage_idx = self.dev2storage[dev_idx]
            bus_idx = self.dev2bus[dev_idx]
            P_total[bus_idx] += Ps['storage'][storage_idx]
            Q_total[bus_idx] += Qs['storage'][storage_idx]

        return P_total, Q_total

    def _get_reactive_injection_storage(self, P_storage, Q_storage):
        """
        Return the reactive power injection of each storage unit.

        This function returns the reactive power injection of each storage
        unit, so that the point (P, Q) is within the operating region of the
        storage unit and all constraints are satisfied. That is, if Q is
        outside of the operating region, it gets simply truncated (i.e. mapped
        onto its maximum value, given a value for P).

        :param P_storage: the real power injection of each storage unit (MW).
        :param Q_storage: the desired reactive power setpoint of each storage
        unit (MVAr).
        :return: the final reactive power injection at each bus (MVAr).
        """

        Q = []
        for dev_idx in self.type2dev['storage']:
            rule = self.pq_rules[dev_idx]
            storage_idx = self.dev2storage[dev_idx]

            p = np.abs(P_storage[storage_idx])
            q = Q_storage[storage_idx]

            # Check that Q is not over its upper limit.
            if q > rule.qmax:
                q = rule.qmax

            # Check that Q is not below its lower limit.
            if q < rule.qmin:
                q = rule.qmin

            # Check that Q is not above the leading slope.
            lead_slope, lead_off = rule.lead_limit
            if q > lead_slope * p + lead_off:
                q = lead_slope * p + lead_off

            # Check that Q is not below the lagging slope.
            lag_slope, lag_off = rule.lag_limit
            if q < lag_slope * p + lag_off:
                q = lag_slope * p + lag_off

            Q.append(q)

        return Q

    def _get_generator_injection_points(self, P_gen):
        """
        Return the final (P, Q) injection points for each distributed generator.

        This function returns the real and reactive power injections for each
        distributed generator, so that the P's are as close as possible to the
        specified ones, while having each (p, q) injection point within the
        feasible region of each generator.

        :param P_gen: the desired real power injection of each generator (MW).
        :return P: the final feasible real power injection of each generator (MW).
        :return Q: the final feasible reactive power injection of each
        generator (MVAr).
        """

        P, Q = [], []
        for dev_idx in self.type2dev['gen']:
            gen_idx = self.dev2gen[dev_idx]
            p, q = self._get_single_gen_injection_point(P_gen[gen_idx], dev_idx)
            P.append(p)
            Q.append(q)
        return P, Q


    def _get_single_gen_injection_point(self, p_gen, dev_idx):
        """
        Return the final (p, q) injection point for a specified generator.

        This function returns the real and reactive power injections of a
        distributed generator. It is computed so that the (p, q) point lies
        within the operating region of the generator, while keeping a constant
        ratio q/p (i.e. constant power factor) and maximizing the real power
        injection p (i.e. to have it as close to the desired value as
        possible).

        :param p_gen: the desired real power injection of the generator (MW).
        :param gen_idx: the device index corresponding to the generator.
        :return: the power injection point (p, q) (MW, MVAr) of the specified
        generator.
        """

        # Get the region of operation of the generator.
        rule = self.pq_rules[dev_idx]

        # Compute the reactive power injection needed to keep a constant power
        # factor.
        p = p_gen

        ### Check that P respects operating constraints. ###
        # Case 1: P is above its upper limit.
        try:
            if p > rule.pmax:
                p = rule.pmax
        except AttributeError:
            pass

        # Case 2: P is under its lower limit.
        try:
            if p < rule.pmin:
                p = rule.pmin
        except AttributeError:
            pass

        ### Check that Q respects operating constraints. ###

        # Compute Q to keep a constant power factor.
        q = p_gen * rule.qp_ratio

        # Case 1: Q is above its upper limit.
        try:
            if q > rule.qmax:
                q = rule.qmax
        except AttributeError:
            pass

        # Case 2: Q is under its lower limit.
        try:
            if q < rule.qmin:
                q = rule.qmin
        except AttributeError:
            pass

        # Case 3: Q is above the leading slope.
        try:
            lead_slope, lead_off = rule.lead_limit
            if q > lead_slope * p + lead_off:
                p = lead_off / (rule.qp_ratio - lead_slope)
                q = p * rule.qp_ratio
        except AttributeError:
            pass

        # Case 4: Q is below the lagging slope.
        try:
            lag_slope, lag_off = rule.lag_limit
            if q < lag_slope * p + lag_off:
                p = lag_off / (rule.qp_ratio - lag_slope)
                q = p * rule.qp_ratio
        except AttributeError:
            pass

        return p, q

    def _get_reactive_injection_load(self, P_load):
        """
        Return the reactive power injection of loads with constant power factor.

        This function returns the reactive power injection of all passive
        loads, assuming that each one maintains a constant power factor. The
        ratio Q/P is extracted from a "PowerCapabilities" object.

        :param P_load: a list of all real power injections at load devices (MW).
        :return: the reactive power injection at each passive load (MVAr).
        """

        Q_load = []
        for dev_idx in self.type2dev['load']:
            load_idx = self.dev2load[dev_idx]
            power_factor = self.pq_rules[dev_idx].qp_ratio
            q = P_load[load_idx] * power_factor
            Q_load.append(q)
        return Q_load

    def _manage_storage(self, desired_alphas):
        """
        Return P and delta_soc for each storage unit, based on the action chosen.

        This function returns the real power injection into the grid and the
        amount of energy leaving/entering the storage unit (MWh). This is
        computed to have a charging rate as close as possible as the one
        chosen by the DSO during the action decision, while keeping storage
        constraints satisfied (max charging rate and max SoC).

        :param desired_alphas: the desired charging rate of each storage unit
        (MW).
        :return P_storage: the real power injection into the grid of each SU (
        MW).
        :return deltas_soc: the energy being transferred from/to the SU (MWh).
        """

        P_storage = []
        deltas_soc = []

        # Get the device indices of all storage units.
        storage_dev_indices = self.type2dev['storage']

        for dev_idx in storage_dev_indices:

            # Get the row number to index the case['storage'] array.
            storage_idx = self.dev2storage[dev_idx]

            # Get P, delta_SoC for a single storage unit.
            p, delta_soc = self._manage_single_su(desired_alphas[storage_idx],
                                                  storage_idx)
            P_storage.append(p)
            deltas_soc.append(delta_soc)

        return P_storage, deltas_soc

    def _manage_single_su(self, desired_alpha, storage_idx):
        """
        Return the P and delta_SoC of the specified storage unit.

        This function returns the real power injection into the network and
        the amount of energy (MWh) leaving/entering the storage unit, so that the
        charging rate is as close as possible to the desired_alpha,
        while satisfying storage constraints (max charging rate, max SoC).

        :param desired_alpha: desired (dis)charging rate for the storage unit
        (MW).
        :param storage_idx: the index of the storage unit.
        :return P: the real power injection into the network from the SU (MW).
        :return delta_soc: the amount of energy moved from/to the SU (MWh).
        """

        eff = self.storage[storage_idx, Simulator.STORAGE_H['EFF']]
        max_rate = self.storage[storage_idx, Simulator.STORAGE_H['PMAX']]
        max_soc = self.max_soc[storage_idx]

        # Compute the real power injection in the network and the energy
        # getting stored in the storage unit.

        # Truncate desired charging rate if above upper limit.
        if desired_alpha > 0.:
            max_alpha = np.minimum(desired_alpha, max_rate)
        else:
            max_alpha = - np.minimum(- desired_alpha, max_rate)

        # Get energy being transferred.
        if desired_alpha > 0.:
            delta_soc = Simulator.DELTA_T * (1 + eff) * max_alpha / 2.
        else:
            delta_soc = Simulator.DELTA_T * max_alpha

        # Check that SoC constraints are satisfied and modify the charging
        # rate if it is.
        # Case 1: upper limit.
        if self.SoC[storage_idx] + delta_soc > max_soc:
            delta_soc = max_soc - self.SoC[storage_idx]
            max_alpha = 2 * delta_soc / (Simulator.DELTA_T * (1 + eff))

        # Case 2: lower limit.
        elif self.SoC[storage_idx] + delta_soc < 0.:
            delta_soc = self.SoC[storage_idx]
            max_alpha = delta_soc / Simulator.DELTA_T

        # Get the real power injection in the network.
        if desired_alpha > 0.:
            P = - max_alpha
        else:
            P = - 2 * eff * max_alpha / (1 + eff)

        return P, delta_soc

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
        real_p = np.real(complex_rhs[1:]) - P # skip slack bus.

        # Equations involving real variables and reactive power injections.
        real_q = np.imag(complex_rhs[1:]) - Q # skip slack bus.

        # Equation fixing the voltage magnitude at the slack bus.
        slack_magn = [np.absolute(V[0]) - self.V_magn_slack]

        # Equation fixing the voltage phase angle to be 0.
        slack_phase = [np.angle(V[0])]

        # Stack equations made of only real variables on top of each other.
        real_equations = np.hstack((real_p, real_q, slack_magn, slack_phase))

        return real_equations

    def _solve_pfes(self, P_bus, Q_bus):
        """
        Solve the power flow equations and return V, P, Q for each bus.

        This function solves the power flow equations of the network. If no
        solution is found, a ValueError is raised and a message displayed. The
        real and reactive power injections at the slack bus are then retrieved
        from the solution V, and the nodal vectors V, P, Q are reconstructed.

        :param P_bus: an (N-1) vector of real power injection (except slack) (
        MW).
        :param Q_bus: an (N-1) vector of reactive power injection (except
        slack) (MVAr).
        :return: the N vectors of nodal V (p.u.), P (MW), and Q (MVAr).
        """

        # Initialize V to represent v_ij = 1 exp(j 0).
        init_v = np.array([self.V_magn_slack] + [1] * (self.N_bus - 1)
                          + [0] * self.N_bus)

        # Transform P, Q injections into p.u. values.
        P_bus_pu = P_bus[1:] / self.baseMVA # skip slack bus.
        Q_bus_pu = Q_bus[1:] / self.baseMVA # skip slack bus.

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
        p_slack = np.real(s_slack)
        q_slack = np.imag(s_slack)

        # Form the nodal P and Q vectors and convert from p.u. to MW and MVAr.
        P = np.hstack((p_slack, P_bus_pu)) * self.baseMVA
        Q = np.hstack((q_slack, Q_bus_pu)) * self.baseMVA

        return V, P, Q

    def _get_current_from_voltage(self, i, j):
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
        v_f, v_t = self.V[i], self.V[j]

        # Compute the complex current in the branch, as seen from node i (p.u.).
        current = np.absolute(tap) ** 2 * (y_s + y_shunt) * v_f - \
                  np.conjugate(tap) * y_s * v_t

        return current


class PowerCapabilities(object):
    """
    This class implements the region of allowed injection points for a device.

    Its attributes include:
        1. description: a string describing the electric device,
        2. qp_ratio: a power ratio (for load and storage devices),
        3. parameters defining a polyhedron (for generator devices):
            3.1. pmin, pmax: bounds on real power injection,
            3.2. qmin, qmax: bounds on reactive power injection,
            3.3. lag_limit, lead_limit: (slope, offset) pairs, each defining a
                                        line in the 2D plane.
    """

    def __init__(self, description='unknown device'):
        self.description = description

    @property
    def qp_ratio(self):
        try:
            return self._qp
        except AttributeError:
            raise AttributeError('The "qp_ratio" is not set for %s.'
                                 % self.description)

    @qp_ratio.setter
    def qp_ratio(self, value):
        self._qp = value

    @property
    def pmin(self):
        try:
            return self._pmin
        except AttributeError:
            raise AttributeError("'pmin' is not set for '%s'." % self.description)

    @pmin.setter
    def pmin(self, value):
        self._pmin = value

    @property
    def pmax(self):
        try:
            return self._pmax
        except AttributeError:
            raise AttributeError("'pmax' is not set for '%s'." % self.description)

    @pmax.setter
    def pmax(self, value):
        self._pmax = value

    @property
    def qmin(self):
        try:
            return self._qmin
        except AttributeError:
            raise AttributeError("'qmin' is not set for '%s'." % self.description)

    @qmin.setter
    def qmin(self, value):
        self._qmin = value

    @property
    def qmax(self):
        try:
            return self._qmax
        except AttributeError:
            raise AttributeError("'qmax' is not set for '%s'." % self.description)

    @qmax.setter
    def qmax(self, value):
        self._qmax = value

    @property
    def lag_limit(self):
        try:
            return (self._lag_slope, self._lag_offset)
        except AttributeError:
            raise AttributeError(
                "'lag_limit' is not set for '%s'." % self.description)

    @lag_limit.setter
    def lag_limit(self, params):
        dQ = params["QC2MIN"] - params["QC1MIN"]
        dP = params["PC2"] - params["PC1"]
        self._lag_slope = dQ / dP
        self._lag_offset = params["QC1MIN"] - dQ / dP * params["PC1"]

    @property
    def lead_limit(self):
        try:
            return (self._lead_slope, self._lead_offset)
        except AttributeError:
            raise AttributeError("'lead_limit' is not set for '%s'." % self.description)

    @lead_limit.setter
    def lead_limit(self, params):
        dQ = params["QC2MAX"] - params["QC1MAX"]
        dP = params["PC2"] - params["PC1"]
        self._lead_slope = dQ / dP
        self._lead_offset = params["QC1MAX"] - dQ / dP * params["PC1"]

