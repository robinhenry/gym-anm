"""The base class for MPC-based policies."""
import cvxpy as cp
import numpy as np

from gym_anm.simulator.components import Load, StorageUnit, Generator, RenewableGen


class MPCAgent(object):
    """
    A base class for an agent that solves a MPC DC Optimal Power Flow.

    This agent accesses the full state of the distribution network at time :math:`t` and
    then solves an N-stage optimization problem. The optimization problem solved
    is an instance of a Direct Current (DC) Optimal Power Flow problem.

    The construction of the DC OPF problem relies on 3 assumptions:

    1. transmission lines are lossless, i.e., :math:`r_{ij} = 0, \\forall e_{ij} \in \mathcal E,`
    2. the difference between adjacent bus voltage angles is small, i.e.,
       :math:`\\angle V_i = \\angle V_j, \\forall e_{ij} \in \mathcal E`,
    3. nodal voltage magnitude are close to unity, i.e., :math:`|V_{i}| = 1, \\forall i \in \mathcal N`.

    The construction of the N-stage optimization problem relies
    on two additional assumptions:

    All sub-classes must implement the :py:func`forecast()` method to provide forecasts
    of demand and generation over the optimization horizon :math:`[t+1,t+N]`.

    All values are used in per-unit.
    """

    def __init__(self, simulator, action_space, gamma, safety_margin=0.9, planning_steps=1):
        """
        Parameters
        ----------
        simulator : :py:class:`gym_anm.simulator.simulator.Simulator`
            The electricity distribution network simulator.
        action_space : :py:class:`gym.spaces.Box`
            The action space of the environment (used to clip actions).
        gamma : float
            The discount factor in [0, 1].
        safety_margin : float, optional
            The safety margin constant :math:`\\beta` in [0, 1], used to further
            constraint the power flow on each transmission line, thus
            likely accounting for the error introduced in the DC approximation.
        planning_steps : int, optional
            The number (N) of stages (time steps) taken into account in the
            optimization problem.
        """

        # Global parameters.
        self.safety_margin = safety_margin
        self.baseMVA = simulator.baseMVA
        self.lamb = simulator.lamb
        self.action_space = action_space
        self.planning_steps = planning_steps
        self.gamma = gamma

        # Information about all sets (buses, branches, etc.).
        self.n_bus = simulator.N_bus
        self.n_dev = simulator.N_device
        self.n_branch = len(simulator.branches)
        self.delta_t = simulator.delta_t
        self.n_gen = simulator.N_non_slack_gen + 1
        self.n_des = simulator.N_des
        self.n_load = simulator.N_load
        self.n_rer = simulator.N_gen_rer
        self.load_ids = [i for i, d in simulator.devices.items() if isinstance(d, Load)]
        self.gen_ids = [i for i, d in simulator.devices.items() if isinstance(d, Generator)]
        self.non_slack_gen_ids = [
            i for i, d in simulator.devices.items() if isinstance(d, Generator) and not d.is_slack
        ]
        self.gen_rer_ids = [i for i, d in simulator.devices.items() if isinstance(d, RenewableGen)]
        self.des_ids = [i for i, d in simulator.devices.items() if isinstance(d, StorageUnit)]
        self.branch_ids = list(simulator.branches.keys())
        self.device_ids = list(simulator.devices.keys())
        self.bus_ids = list(simulator.buses.keys())
        self.slack_dev_id = [
            i
            for i in self.device_ids
            if i not in self.des_ids and i not in self.non_slack_gen_ids and i not in self.load_ids
        ][0]

        # Define a mapping from the device to the buses they are connected to
        # (using the simulator internal indices).
        self.dev_to_bus = {i: d.bus_id for i, d in simulator.devices.items()}

        # Define a mapping from `bus_id` to [0, n_bus], where n_bus is the actual
        # number of buses in the network. This is useful if the bus indices
        # skip some numbers (e.g., original bus indices where [0, 1, 3]).
        self.bus_id_mapping = {}
        for i, bus_id in enumerate(simulator.buses.keys()):
            self.bus_id_mapping[bus_id] = i

        # Define a similar mapping for the devices.
        self.dev_id_mapping = {}
        for i, dev_id in enumerate(simulator.devices.keys()):
            self.dev_id_mapping[dev_id] = i

        # Empty lists to store all single-timestep optimization variables.
        self.V_bus_ang_vars = []
        self.P_dev_vars = []

        # Define time-varying parameters.
        self.P_load_forecast = cp.Parameter((self.n_load, self.planning_steps))
        self.P_gen_forecast = cp.Parameter((self.n_gen - 1, self.planning_steps), nonneg=True)
        self.init_soc = cp.Parameter(self.n_des, nonneg=True)

        # Define constants.
        bus_ids = [self.bus_id_mapping[i] for i in self.bus_ids]
        self.B_bus = simulator.Y_bus.imag[bus_ids, :][:, bus_ids].toarray()
        self.branch_rate = [br.rate for br in simulator.branches.values()]
        self.P_gen_min = [g.p_min for g in simulator.devices.values() if isinstance(g, Generator) and not g.is_slack]
        self.P_gen_max = [g.p_max for g in simulator.devices.values() if isinstance(g, Generator) and not g.is_slack]
        self.P_des_min = [d.p_min for d in simulator.devices.values() if isinstance(d, StorageUnit)]
        self.P_des_max = [d.p_max for d in simulator.devices.values() if isinstance(d, StorageUnit)]
        self.soc_min = [d.soc_min for d in simulator.devices.values() if isinstance(d, StorageUnit)]
        self.soc_max = [d.soc_max for d in simulator.devices.values() if isinstance(d, StorageUnit)]
        self.des_eff = [d.eff for d in simulator.devices.values() if isinstance(d, StorageUnit)]

        # Indicate the optimization problem has not been constructed yet. This is
        # done during the first call of `act()`.
        self.dc_opf = None

    def _create_p_bus_expressions(self, P_dev):
        """
        Define :math:`P_i^{(bus)} = sum_d P_d^{(dev)}` expressions.

        Returns
        -------
        list of cvxpy.Expression
            The active power injection at each bus expressed in terms of the
            active power injection of its devices.
        """
        P_bus = [0.0] * self.n_bus
        for d in self.device_ids:
            i = self.dev_to_bus[d]
            P_bus[self.bus_id_mapping[i]] += P_dev[self.dev_id_mapping[d]]

        return P_bus

    def _create_p_branch_expressions(self, V_bus_ang):
        """
        Define :math:`P_{ij} = B_{ij} * (V_ang[i] - V_ang[j])` expressions.

        Returns
        -------
        list of cvxpy.Expression
            The branch active power flow expressed in terms of the nodal voltage
            phase angle variables.
        """
        P_branch = []
        for i, j in self.branch_ids:
            k = self.bus_id_mapping[i]
            l = self.bus_id_mapping[j]
            c = self.B_bus[k, l] * (V_bus_ang[k] - V_bus_ang[l])
            P_branch.append(c)

        return P_branch

    def _create_optimization_problem(self):
        """
        Create the N-stage cvxpy optimization problem.

        Returns
        -------
        problem : cvxpy.Problem
            The N-stage convex optimization problem.
        """

        objective = 0.0
        constraints = []

        # Variable coupled between different time steps.
        soc = self.init_soc

        for i in range(self.planning_steps):

            # Extract forecasted values for timestep t+1+i.
            P_load_forecast = self.P_load_forecast[:, i]
            P_gen_forecast = self.P_gen_forecast[:, i]

            # Create a new single-step optimization problem coupled with the
            # previous time step.
            obj, consts, optim_vars, soc = self._single_step_optimization_problem(P_load_forecast, P_gen_forecast, soc)
            objective += self.gamma**i * obj
            constraints += consts

            # Store the optimization variables.
            self.P_dev_vars.append(optim_vars["P_dev"])
            self.V_bus_ang_vars.append(optim_vars["V_bus_ang"])

        # 3. Construct the final N-stage optimization problem.
        problem = cp.Problem(cp.Minimize(objective), constraints)

        # 4. Extract the final optimization variables (from the first stage).
        self.P_dev = self.P_dev_vars[0]
        self.V_bus_ang = self.V_bus_ang_vars[0]

        return problem

    def _single_step_optimization_problem(self, P_load_forecast, P_gen_forecast, soc_prev):
        """
        Define a single-stage instance of the DC OPF optimization problem.

        Parameters
        ----------
        P_load_forecast : cvxpy.Expression
            The vector of forecasted load active power injections.
        P_gen_forecast : cvxpy.Expression
            The vector of forecasted maximum active power generation from non-slack
            generators.
        soc_prev : cvxpy.Expression
            The vector of current state of charge of DES units.

        Returns
        -------
        obj : cvxpy.Expression
            The objective to be minimized.
        constraints : list of cvxpy.Constraint
            A list of constraints.
        optim_var : dict of {str : cvxpy.Variable or cvxpy.Expression}
            The optimization variables, with keys {'V_bus_ang', 'P_dev', 'P_bus',
            'P_branch'}.
        new_socs : list of cvxpy.Expression
            The state of charge of the DES units at the next time step, expressed
            in terms of the optimization variables.
        """

        # 1. Create optimization variables for this timestep.
        V_bus_ang = cp.Variable(self.n_bus)
        P_dev = cp.Variable(self.n_dev)

        # 2. Define P_bus and P_branch expressions.
        P_bus = self._create_p_bus_expressions(P_dev)
        P_branch = self._create_p_branch_expressions(V_bus_ang)

        # 3. Construct the constraints.
        constraints = []

        # P_bus[i] = \sum_{ij} B_{ij} (V_ang[i] - V_ang[j]).
        a = []
        for i in self.bus_ids:
            c = 0.0
            for j, k in self.branch_ids:
                l = self.bus_id_mapping[j]
                m = self.bus_id_mapping[k]
                if j == i:
                    c += self.B_bus[l, m] * (V_bus_ang[l] - V_bus_ang[m])
                elif k == i:
                    c += self.B_bus[m, l] * (V_bus_ang[m] - V_bus_ang[l])
            a.append(c)
        constraints += _make_list_eq_constraints(a, P_bus)

        # P_load(t+1) = P_load_forecast
        c = []
        for l in self.load_ids:
            c.append(P_dev[self.dev_id_mapping[l]])
        constraints += _make_list_eq_constraints(c, P_load_forecast)

        # P_min <= P_gen <= P_max_ (for non-slack generators).
        c = []
        for g in self.non_slack_gen_ids:
            c.append(P_dev[self.dev_id_mapping[g]])
        constraints += _make_list_le_constraints(self.P_gen_min, c)
        constraints += _make_list_le_constraints(c, self.P_gen_max)

        # P_min <= P_des <= P_max (for DES units).
        c = []
        for des in self.des_ids:
            c.append(P_dev[self.dev_id_mapping[des]])
        constraints += _make_list_le_constraints(self.P_des_min, c)
        constraints += _make_list_le_constraints(c, self.P_des_max)

        # P <= P_gen_forecast (for non-slack generators).
        c = []
        for gen in self.non_slack_gen_ids:
            c.append(P_dev[self.dev_id_mapping[gen]])
        constraints += _make_list_le_constraints(c, P_gen_forecast)

        # P >= (soc_{t-1} - soc_max) / (delta_t * eff)
        # P <= (soc_{t-1} - soc_min) * eff / delta_t
        new_socs, c = [], []
        for i, des in enumerate(self.des_ids):
            p_charging = cp.Variable(nonneg=True)
            p_discharging = cp.Variable(nonneg=True)
            delta_soc_charging = p_charging * self.delta_t * self.des_eff[i]
            delta_soc_discharging = -p_discharging * self.delta_t / self.des_eff[i]
            new_soc = soc_prev[i] + delta_soc_charging + delta_soc_discharging
            new_socs.append(new_soc)
            c.append(P_dev[self.dev_id_mapping[des]] == p_discharging - p_charging)

        constraints += c
        constraints += _make_list_le_constraints(self.soc_min, new_socs)
        constraints += _make_list_le_constraints(new_socs, self.soc_max)

        # - \pi <= V_angle <= \pi
        constraints += _make_list_le_constraints([-np.pi] * self.n_bus, V_bus_ang)
        constraints += _make_list_le_constraints(V_bus_ang, [np.pi] * self.n_bus)

        # V_angle[0] = 0
        constraints.append(V_bus_ang[self.dev_id_mapping[self.slack_dev_id]] == 0.0)

        # 4. Construct the objective function.
        cost_1 = 0.0
        for gen in self.gen_ids:
            if gen not in self.gen_rer_ids:
                cost_1 += P_dev[self.dev_id_mapping[gen]]

        cost_2 = 0.0
        for p, rate in zip(P_branch, self.branch_rate):
            cost_2 += cp.maximum(0, cp.abs(p) - self.safety_margin * rate)

        obj = cost_1 + self.lamb * cost_2

        # 5. Optimization variable dictionary to return.
        optim_vars = {"V_bus_ang": V_bus_ang, "P_dev": P_dev, "P_bus": P_bus, "P_branch": P_branch}

        return obj, constraints, optim_vars, new_socs

    def act(self, env):
        """
        Select an action by solving the N-stage DC OPF.

        Parameters
        ----------
        env : py:class:`gym_anm.ANMEnv`
            The :code:`gym-anm` environment.

        Returns
        -------
        numpy.ndarray
            The action vector to apply in the environment.
        """
        P_load_forecasts, P_gen_forecasts = self.forecast(env)
        a = self._solve(env.simulator, P_load_forecasts, P_gen_forecasts)

        # Clip the actions, which are sometime beyond the space by a tiny
        # amount due to precision errors in the optimization problem
        # solution (e.g., of the order of 1e-10).
        a = np.clip(a, self.action_space.low, self.action_space.high)

        return a

    def forecast(self, env):
        """
        Forecast the demand and generation over the optimization horizon.

        This method must be implemented by all sub-classes of :py:class:`MPCAgent`.
        It gets called in :py:func:`act()` and must return the predictions of
        future load demand, :math:`\\tilde P_{l,k}^{(dev)}`, and maximum generation
        at non-slack generators, :math:`\\tilde P_{g,k}^{(max)}`.

        Parameters
        ----------
        env : py:class:`gym_anm.ANMEnv`
            The :code:`gym-anm` environment.

        Returns
        -------
        P_load_forecast : array_like
            A (N_load, N) array of forecasted load power injections (<0), where N
            is the length of the optimization horizon. The rows should be ordered
            in increasing order of device ID.
        P_gen_forecast : array_like
            A (N_gen-1, N) array of forecasted maximum generation from non-slack
            generators, where N is the length of the optimization horizon. The rows
            should be ordered in increasing order of device ID.
        """
        raise NotImplementedError()

    def _solve(self, simulator, load_forecasts, gen_forecasts):
        """Select an action by solving the N-stage DC OPF."""

        # Initialize the optimization problem during the first iteration.
        if self.dc_opf is None:
            self.dc_opf = self._create_optimization_problem()

        # Update the time-varying parameters (fixed values).
        self._update_parameters(simulator, load_forecasts, gen_forecasts)

        # Solve the DC OPF (convex program).
        self.dc_opf.solve()
        if self.dc_opf.status != "optimal":
            print("OPF problem is " + self.dc_opf.status)

        # Extract the control variables (scale from p.u. to MW or MVAr).
        P_gen = [self.P_dev.value[self.dev_id_mapping[d]] * self.baseMVA for d in self.non_slack_gen_ids]
        Q_gen = [0.0] * len(P_gen)
        P_des = [self.P_dev.value[self.dev_id_mapping[d]] * self.baseMVA for d in self.des_ids]
        Q_des = [0.0] * len(P_des)

        return np.concatenate((P_gen, Q_gen, P_des, Q_des))

    def _update_parameters(self, simulator, P_load_forecasts, P_gen_forecasts):
        """
        Update the time-dependent fixed values of the optimization problem.

        Parameters
        ----------
        simulator : py:class:`gym_anm.simulator.Simulator`
            The power grid simulator of the environment.
        P_load_forecasts : numpy.ndarray
            A (N_load, N) array of forecasted demand values over the optimization
            horizon [t+1, t+N]. The rows should be ordered in increasing order of
            device index.
        P_gen_forecasts : numpy.ndarray
            A (N_gen-1, N) array of forecasted maximum generation from non-slack
            generators over the optimization horizon [t+1, t+N]. The rows should
            be ordered in increasing order of device index.
        """

        # Update the parameters of the optimization problem with the
        # forecasted values.
        self.P_load_forecast.value = np.array(P_load_forecasts)
        self.P_gen_forecast.value = np.array(P_gen_forecasts)

        # Set the initial state of charge of DES units.
        self.init_soc.value = [simulator.state["des_soc"]["pu"][i] for i in self.des_ids]


def _make_list_eq_constraints(a, b):
    """Create a list of cvxpy equality constraints from two lists."""
    if isinstance(a, list):
        n = len(a)
    elif isinstance(b, list):
        n = len(b)
    else:
        raise ValueError()
    return [a[i] == b[i] for i in range(n)]


def _make_list_le_constraints(a, b):
    """Create a list of cvxpy less than or equal (<=) equality constraints."""
    if isinstance(a, list):
        n = len(a)
    elif isinstance(b, list):
        n = len(b)
    else:
        raise ValueError()
    return [a[i] <= b[i] for i in range(n)]
