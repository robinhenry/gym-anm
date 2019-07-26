import numpy as np
import warnings
import scipy.optimize as optimize

class WindGenerator(object):
    def __init__(self, p_init=10):
        self.p = p_init

    def __iter__(self):
        return self

    def next(self):
        if self.p >= 0.:
            p = self.p
        else:
            p = - self.p

        self.p += np.random.rand() * 2
        return p

class LoadGenerator(object):
    def __init__(self, p_init=10):
        self.p = p_init

    def __iter__(self):
        return self

    def next(self):
        if self.p <= 0.:
            p = self.p
        else:
            p = - self.p

        self.p += np.random.rand() * 2
        return p


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
                   'QC1MAX', 'QC2MIN', 'QC2MAX', 'RAMP_AGC', 'RAMP_10',
                   'RAMP_30', 'RAMP_Q', 'APF']
    GEN_H = dict(zip(headers_gen, range(len(headers_gen))))

    headers_branch = ['F_BUS', 'T_BUS', 'BR_R', 'BR_X', 'BR_B', 'RATE_A',
                      'RATE_B', 'RATE_C', 'TAP', 'SHIFT', 'BR_STATUS',
                      'ANGMIN', 'ANGMAX']
    BRANCH_H = dict(zip(headers_branch, range(len(headers_branch))))

    headers_storage = ['BUS_I', 'MAX_C', 'EFF', 'MAX_R', 'PMAX', 'QMAX',
                       'QMIN', 'PC1', 'PC2', 'QC1MIN', 'QC1MAX', 'QC2MIN',
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

        # Create mapping dictionaries between device and bus indices.
        self._init_build_device_bus_mapping()

        # Number of elements in all sets.
        self.N_bus = self.buses.shape[0]  # Number of buses.
        self.N_branch = self.branches.shape[0] # Number of transmission lines.
        self.N_device = self.gen.shape[0] + self.storage.shape[0] # of devices.
        self.N_storage = self.storage.shape[0] # Number of storage units.

        # Create restriction on (P, Q) injection points of devices.
        self._init_build_pq_rules()

        # Build the nodal admittance matrix.
        #self._init_build_admittance_matrix()

        # Get bounds on current magnitudes in all branches.
        self.Imax = self.branches[:, Simulator.BRANCH_H['RATE_A']] / self.baseMVA

        # Get bounds on voltage magnitude at all buses, except the slack bus.
        self.Vmin = self.buses[1:, Simulator.BUS_H['VMIN']]
        self.Vmax = self.buses[1:, Simulator.BUS_H['VMAX']]

        # Get voltage magnitude set-point at slack bus.
        self.V_slack = self.gen[0, Simulator.GEN_H['VG']]

        # Check that the set-point voltage magnitude at the slack bus
        # seems coherent.
        if (self.V_slack < 0.5) or (self.V_slack > 1.5):
            warnings.warn("Warning: voltage magnitude (" + str(self.V_slack)
                          + ") at the slack bus does not seem coherent.")

        # Initialize dummmy generators of time series for DGs and passive loads.
        self.wind_gens = [WindGenerator() for _ in self.type2dev['generator']]
        self.load_gens = [LoadGenerator() for _ in self.type2dev['load']]

        # Initialize vectors of injections from each device.
        self.P_dev = np.zeros((self.N_device))
        self.Q_dev = np.zeros((self.N_device))

        # Initialize electric quantities at buses.
        self.V = np.zeros(shape=(self.N_bus,), dtype=np.complex)
        self.I = np.zeros(shape=(self.N_bus,), dtype=np.complex)
        self.P = np.zeros(shape=(self.N_bus,))
        self.Q = np.zeros(shape=(self.N_bus,))

        # Initialize variables linked to curtailment.
        self.P_potential = np.zeros(shape=(self.N_device,))

        # Initialize storage unit variables.
        self.SoC = self.storage[:, Simulator.STORAGE_H['MAX_C']] / 2.
        self.alpha = np.zeros((self.N_device,))

        # Initialize control (action) variables.
        self.P_curt_limit = np.zeros(shape=(self.N_device,))
        self.desired_alpha = np.zeros(shape=(self.N_device,))

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

        :return:
        """

        # Initialize an empty dictionary to create a mapping device -> bus.
        dev2bus = {}
        # Initialize empty dictionary to map device_idx -> storage_idx.
        dev2storage = {}

        # Add indices of generators and loads.
        idx = 0
        for i in range(self.gen.shape[0]):
            dev2bus[i] = int(self.gen[i, 0])
            idx = i

        idx += 1

        # Add indices of storage units.
        for i in range(idx, idx + self.N_storage):
            dev2bus[i] = int(self.storage[i - idx, 0])
            dev2storage[i] = i - idx

        self.dev2bus = dev2bus
        self.dev2storage = dev2storage

        # Initialize an empty dictionary to store lists of indices to easily
        # retrieve all generators / loads / storage units.
        type2dev = {'slack': 0} # initialize with slack bus device.

        # Add distributed generators.
        type2dev['gen'] = np.nonzero(self.gen[:, Simulator.GEN_H['PG']] > 0.)[0]
        type2dev['load'] = np.nonzero(self.gen[:, Simulator.GEN_H['PG']] < 0.)[0]
        type2dev['storage'] = np.arange(idx + 1, idx + 1 + self.N_storage)

        self.type2dev = type2dev

    def _init_build_pq_rules(self):
        """
        Get the region of power injection capabilities for a load or gen.

        This function creates a PowerCapabilities object, for each passive
        load and distributed generator, storing the information needed to
        define the corresponding region of feasible (P, Q) injections.
        """

        self.pq_rules_load = {}
        self.pq_rules_gen = {}
        self.pq_rules_storage = {}

        # Store restrictions for passive load power injections in a dict.
        for load_idx in self.type2dev['load']:
            rule = PowerCapabilities('passive load')

            rule.qp_ratio = self.gen[load_idx, Simulator.GEN_H['QG']] / \
                            self.gen[load_idx, Simulator.GEN_H['PG']]
            self.pq_rules_load[load_idx] = rule

        # Store restrictions for distributed generators in a dict.
        indices = np.concatenate((self.type2dev['gen'], self.type2dev['slack']))
        for gen_idx in indices:
            rule = PowerCapabilities('distributed generator')

            rule.qp_ratio = self.gen[gen_idx, Simulator.GEN_H['QG']] / \
                            self.gen[gen_idx, Simulator.GEN_H['PG']]


            rule.pmin = self.gen[gen_idx, Simulator.GEN_H['PMIN']]
            rule.qmin = self.gen[gen_idx, Simulator.GEN_H['QMIN']]

            if self.gen[gen_idx, Simulator.GEN_H['PMAX']] > 0.:
                rule.pmax = self.gen[gen_idx, Simulator.GEN_H['PMAX']]

            if self.gen[gen_idx, Simulator.GEN_H['QMAX']] > 0.:
                rule.qmax = self.gen[gen_idx, Simulator.GEN_H['QMAX']]

            if self.gen[gen_idx, Simulator.GEN_H['PC1']] \
                != self.gen[gen_idx, Simulator.GEN_H['PC2']]:
                rule.lead_limit = {k: self.gen[gen_idx, Simulator.GEN_H[k]] for
                                   k in ['PC1', 'PC2', 'QC1MAX', 'QC2MAX']}
                rule.lag_limit = {k: self.gen[gen_idx, Simulator.GEN_H[k]] for
                                  k in ['PC1', 'PC2', 'QC1MIN', 'QC2MIN']}

            self.pq_rules_gen[gen_idx] = rule

        # Store restrictions for storage units in a dict.
        for storage_idx in range(self.N_storage):
            rule = PowerCapabilities('storage unit')

            if self.storage[storage_idx, Simulator.STORAGE_H['PMAX']] > 0.:
                rule.pmax = self.storage[storage_idx, Simulator.STORAGE_H['PMAX']]

            rule.qmin = self.storage[storage_idx, Simulator.STORAGE_H['QMIN']]
            rule.qmax = self.storage[storage_idx, Simulator.STORAGE_H['QMAX']]

            if self.storage[storage_idx, Simulator.STORAGE_H['PC1']] \
                != self.storage[storage_idx, Simulator.GEN_H['PC2']]:
                rule.lead_limit = {k: self.storage[storage_idx,
                                                   Simulator.STORAGE_H[k]]
                                   for k in ['PC1', 'PC2', 'QC1MAX', 'QC2MAX']}
                rule.lag_limit = {k: self.storage[storage_idx,
                                                  Simulator.STORAGE_H[k]]
                                  for k in ['PC1', 'PC2', 'QC1MIN', 'QC2MIN']}

            self.pq_rules_storage[storage_idx] = rule

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

        else:
            raise ValueError("The slack bus of the test case must be specified "
                             "as the first bus in the input file. case['bus']["
                             "0, 1] == 3 should be true.")

    def _init_build_admittance_matrix(self):
        """
        Build the nodal admittance matrix of the network.

        This function builds the nodal admittance matrix of the network,
        based on specifications given in the input case file.
        """

        # Initialize an N-by-N empty complex admittance matrix.
        Y_bus = np.zeros((self.N_bus, self.N_bus), dtype=np.complex)

        # Initialize an N_branch-by-N_branch matrix to store tap ratios,
        # series and shunt admittances of branches.
        taps = np.zeros((self.N_branch, self.N_branch), dtype=np.complex)
        shunts = np.zeros((self.N_branch, self.N_branch), dtype=np.complex)
        series = np.zeros((self.N_branch, self.N_branch), dtype=np.complex)

        # Build the complex nodal admittance matrix, iterating over branches.
        for branch in self.branches:

            # Get sending and receiving buses of the branch (0 indexing).
            bus_f = branch[0] - 1
            bus_t = branch[1] - 1

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
            taps[bus_f, bus_t] = tap
            taps[bus_t, bus_f] = 1.
            shunts[bus_f, bus_t] = y_shunt
            shunts[bus_t, bus_f] = y_shunt
            series[bus_f, bus_t] = y_series
            series[bus_t, bus_f] = y_series

        self.Y_bus = Y_bus
        self.taps = taps
        self.shunts = shunts
        self.series = series

        # Separate the real and complex parts of the admittance matrix into
        # two matrices of the same dimensions.
        self.G_bus = np.real(Y_bus)
        self.B_bus = np.imag(Y_bus)


    def _get_transmission_loss(self):
        """
        Return the total real loss due to transmission over the network.

        This function computes the total real loss that is due to non-ideal
        components in the distribution grid, per hour. This is done by summing
        all the real power injection; the resulting value is the difference
        between production and demand.

        :return: the total per-hour real loss due to transmission over the
        network.
        """
        return np.sum(self.P) * Simulator.DELTA_T

    def _get_curtailment_loss(self):
        """
        Return the total real loss due to curtailment of generators.

        This function returns the total energy loss due to curtailment,
        over the network. This is equal to the difference between the
        potential generation (assumed known) and the real generation after
        curtailment.

        :return: the total real loss due to curtailment of generators.
        """
        return np.sum(np.max(self.P_potential - self.P)) * Simulator.DELTA_T

    def _get_storage_loss(self):
        """
        Return the total real loss due to inefficiency of storage units.

        This function returns the total real loss due to losses in storage
        units, over the network. This is equal to the energy loss by each
        storage unit, during charging or discharging.

        :return: the total real loss due to storage units.
        """
        pass

    def _get_penalty(self):
        """
        Return the penalty associated with operation constraints violation.

        This function returns a (big) penalty cost if the system violates
        operation constraints, that is voltage magnitude and line current
        constraints.

        :return: the penalty associated with operation constraints violation.
        """

        # Compute the total voltage constraints violation.
        V_magn = np.absolute(self.V)
        V_penalty = np.sum(np.max(0, V_magn - self.Vmax) \
                           + np.max(0, self.Vmin - V_magn))

        # Compute the total current constraints violation.
        I_magn = np.abs(self.I)
        I_penalty = np.sum(np.max(0, I_magn - self.Imax))

        return Simulator.LAMB * (V_penalty + I_penalty)

    def _get_prod_limit(self, gen_idx):
        """
        Return the generation upper limit set during the last curtailment.

        This function returns the upper limit on real power generation at the
        specified generator device, which was set during the last curtailment
        instructions.

        :param gen_idx: the index of the generator device.
        :return: the upper limit on real power generation at specified bus.
        """
        return self.P_curt_limit[gen_idx]

    def _get_branch_current_magn(self, branch_idx):
        """
        Return the current magnitude on a given transmission line, in per-unit.

        :param branch_idx: the index of the desired transmission line.
        :return: the current magnitude on the given branch.
        """
        return np.absolute(self.I[branch_idx])

    def _get_bus_voltage_magn(self, bus_idx):
        """
        Return the voltage magnitude at a given bus, in per-unit.

        :param bus_idx: index of the desired bus.
        :return: the voltage magnitude at the desired bus.
        """
        return np.absolute(self.V[bus_idx])

    def is_safe(self):
        """
        Check whether the current system state violates operating constraints.

        This function checks whether the current state of the system violates
        either voltage magnitude constraints, or current magnitudes constraints.

        :return: False if any operating constraint is violated; True otherwise.
        """
        penalty = self._get_penalty()

        if penalty > 0.:
            return False
        else:
            return True

    def get_reward(self):
        """
        Return the total reward associated with the current state of the system.

        The reward is computed as a negative sum of transmission
        losses, curtailment losses, (dis)charging losses, and operational
        constraints violation costs.

        :return: the total reward associated with the current system state.
        """

        total_energy_loss = self._get_transmission_loss() \
                            + self._get_curtailment_loss() \
                            + self._get_storage_loss()
        penalty = self._get_penalty()

        return - (total_energy_loss + penalty)

    def transition(self, P_curt_limit, desired_alpha, Q_storage_setpoints):
        """
        Simulates a transition of the system from time t to time (t+1).

        This function simulates a transition of the system after actions were
        taken by the DSO, during the previous time step. The results of these
        decisions then affect the new state of the system.
        """

        ### Manage passive loads. ###
        # 1. Get the real demand at each load.
        P_load = [load.next() for load in self.load_gens]

        # 2. Compute the reactive power injection at each load.
        Q_load = self._get_reactive_injection_load(P_load)

        ### Manage distributed generators. ###
        # 1. Get the potential production at each generator.
        P_potential = [gen.next() for gen in self.wind_gens]

        # 2. Curtail potential production.
        P_curt = np.minimum(P_potential, np.maximum(0, P_curt_limit))

        # 3. Get (P, Q) feasible injection points.
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

        # 2. Solve PFEs and store nodal V, P, Q vectors.
        V, P, Q = self._solve_pfes(P_bus, Q_bus)

        # 3. Compute I in each branch (in each direction) from V.

        ### Compute total reward. ###


    def _get_bus_total_injections(self, Ps, Qs):
        """
        Return the total real and reactive power injection at each buss.

        :param Ps: a dict of lists of real power injection with keys {
        'generator', 'load', 'storage'}.
        :param Qs: a dict of lists of reactive power injection with keys {
        'generator', 'load', 'storage'}.
        :return: the total real and reactive power injection at each bus.
        """

        P_total = np.zeros(shape=(self.N_bus))
        Q_total = np.zeros(shape=(self.N_bus))

        # Iterate over generator devices.
        gen_idx = 0
        for dev_idx in self.type2dev['generator']:
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

        :param P_storage: the real power injection of each storage unit.
        :param Q_storage: the desired
        :return:
        """

        Q = []
        for storage_idx in range(self.N_storage):
            rule = self.pq_rules_storage[storage_idx]

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

        :param P_gen: the desired real power injection of each generator.
        :return P: the final feasible real power injection of each generator.
        :return Q: the final feasible reactive power injection of each generator.
        """

        P, Q = [], []
        for gen_idx in range(len(P_gen)):
            p, q = self._get_single_gen_injection_point(P_gen[gen_idx], gen_idx)
            P.append(p)
            Q.append(q)
        return P, Q


    def _get_single_gen_injection_point(self, p_gen, gen_idx):
        """
        Return the final (p, q) injection point for a specified generator.

        This function returns the real and reactive power injections of a
        distributed generator. It is computed so that the (p, q) point lies
        within the operating region of the generator, while keeping a constant
        ratio q/p (i.e. constant power factor) and maximizing the real power
        injection p (i.e. to have it as close to the desired value as
        possible).

        :param p_gen: the desired real power injection of the generator.
        :param gen_idx: the index of the generator.
        :return: the power injection point (p, q) of the specified generator.
        """

        # Get the region of operation of the generator.
        rule = self.pq_rules_gen[gen_idx]

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

        :param P_load: a list of all real power injections at load devices.
        :return: the reactive power injection at each passive load.
        """

        Q_load = []
        for load_idx in range(len(P_load)):
            power_factor = self.pq_rules_load[load_idx].qp_ratio
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

        :param desired_alphas: the desired charging rate of each storage unit.
        :return P_storage: the real power injection into the grid of each SU.
        :return deltas_soc: the energy being transferred from/to the SU.
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

        :param desired_alpha: desired (dis)charging rate for the storage unit.
        :param storage_idx: the index of the storage unit.
        :return P: the real power injection into the network from the SU.
        :return delta_soc: the amount of energy moved from/to the SU.
        """

        eff = self.storage[storage_idx, Simulator.STORAGE_H['EFF']]
        max_rate = self.storage[storage_idx, Simulator.STORAGE_H['MAX_R']]
        max_soc = self.storage[storage_idx, Simulator.STORAGE_H['MAX_C']]

        # Compute the real power injection in the network and the energy
        # getting stored in the storage unit.

        # Truncate desired charging rate if above upper limit.
        if desired_alpha > 0.:
            max_alpha = np.max(desired_alpha, max_rate)
        else:
            max_alpha = np.max(- desired_alpha, max_rate)

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

        # Re-build complex nodal voltage array.
        V = v[:self.N_bus] + 1.j * v[self.N_bus:]

        # Create complex equations as a matrix product.
        complex_rhs = V * np.dot(Y, V)

        # Equations involving real variables and real power injections.
        real_p = complex_rhs[1:].real() - P # skip slack bus.

        # Equations involving real variables and reactive power injections.
        real_q = complex_rhs[1:].imag() - Q # skip slack bus.

        # Equation fixing the voltage magnitude at the slack bus.
        slack_magn = [np.absolute(V[0]) - self.V_slack]

        # Equation fixing the voltage phase angle to be 0.
        slack_phase = [V[0].angle()]

        # Stack equations made of only real variables on top of each other.
        real_equations = np.hstack((real_p, real_q, slack_magn, slack_phase))

        return real_equations

    def _solve_pfes(self, P_bus, Q_bus):
        """
        Solve the power flow equations and return V, P, Q for each bus.
        """

        init_v = np.array([])

        # Solve the power flow equations of the network.
        sol = optimize.root(self._power_flow_eqs, init_v,
                          args=(self.Y_bus, P_bus, Q_bus),
                          method='hybr', options={'xtol': 1.0e-4})
        if not sol.success:
            raise ValueError('No solution to the PFEs: ', sol.message)
        x = sol.x

        # Re-create the complex bus voltage vector.
        self.V = x[0:self.N_bus] + 1.j * x[self.N_bus:]

        # Compute the complex power injection at the slack bus.
        s_slack =

        # Retrieve the real and reactive power injections at the slack bus (=
        # slack device, since there is only 1 device at the slack bus).

        # Form the nodal P and Q vectors.






    def _get_current_from_voltage(self, i, j):
        """
        Return the current sent on transmission line (i, j) at node i.

        This function computes the line complex current from bus i to bus j,
        based on voltage values. This is the current value as it leaves node
        i. Note that I_ij is different from I_ji.

        :param i: the sending end of the line.
        :param j: the receiving end of the line.
        :return: the complex current on the desired transmission line.
        """

        # Get the characteristics of the transmission line and transformer.
        tap = self.taps[i, j]
        y_s = self.series[i, j]
        y_shunt = self.shunts[i, j]
        v_f, v_t = self.V[i], self.V[j]

        # Compute the complex current in the branch, as seen from node i.
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
        assert type(params) is dict
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
        assert type(params) is dict
        dQ = params["QC2MAX"] - params["QC1MAX"]
        dP = params["PC2"] - params["PC1"]
        self._lead_slope = dQ / dP
        self._lead_offset = params["QC1MAX"] - dQ / dP * params["PC1"]

