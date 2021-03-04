import cvxpy as cp
import numpy as np

from . import MPCAgent


class MPCAgentANM6Easy(MPCAgent):
    """
    A deterministic Model Predictive Control agent for the ANM6Easy environment.

    This agent solves the :code:`ANM6Easy` environment as a Model Predictive
    Control (MPC) problem. It implements a near-optimal policy, by accessing
    the full state of the distribution network, as well as the future demand
    and maximum generation from loads and distributed generators, respectively.

    This agent improves the base class :py:class:`gym_anm.agents.dc_opf.DCOPFAgent` by removing the
    assumptions that the demand of loads and the maximum generation of non-slack
    generators is constant during :math:`[t, t+N]`.

    The resulting policy is optimal, if we assume that N optimization stages are
    used, with :math:`N \\to \infty`, and that the approximation errors introduced when
    casting the original AC OPF problem into a linear DC OPF problem are
    negligible.

    All values are used in per-unit.
    """

    def __init__(self, simulator, action_space, gamma, safety_margin=0.9,
                 planning_steps=1):
        """
        Parameters
        ----------
        simulator : :py:class:`gym_anm.simulator.simulator.Simulator`
            The electricity distribution network simulator.
        action_space : gym.spaces.Box
            The action space of the environment (used to clip actions).
        gamma : float
            The discount factor in [0, 1].
        safety_margin : float, optional
            The safety margin constant :math:`\\beta` in [0, 1], used to further
            constraint the power flow on each transmission line, thus
            likely accounting for the error introduced in the DC approximation.
        planning_steps : int
            The number (N) of stages (time steps) taken into account in the
            optimization problem.
        """

        super().__init__(simulator, action_space, gamma, safety_margin,
                         planning_steps)

        # Overwrite the variables storing the future loads and max generations.
        self.P_load_prev = cp.Parameter((self.n_load, self.planning_steps))
        self.P_gen_pot_prev = cp.Parameter((self.n_rer, self.planning_steps),
                                           nonneg=True)

    def _create_optimization_problem(self):

        objective = 0.
        constraints = []

        # Fixed variables, provided as constants at the start of the
        # optimization.
        P_load = self.P_load_prev
        P_pot = self.P_gen_pot_prev

        # Variables coupled between different time steps.
        soc = self.soc_prev

        for i in range(self.planning_steps):

            # Create a new optimization problem coupled with the previous
            # timestep. The main difference with the base class is here: the
            # p_load and p_pot are taken from known time series, and not assumed
            # constants anymore.
            obj, consts, optim_vars, soc = \
                self._single_step_optimization_problem(P_load[:, i], soc,
                                                       P_pot[:, i])
            objective += self.gamma ** i * obj
            constraints += consts

            # Store the optimization variables.
            self.P_dev_vars.append(optim_vars['P_dev'])
            self.V_bus_ang_vars.append(optim_vars['V_bus_ang'])

        # 3. Construct the final multi-step optimization problem.
        problem = cp.Problem(cp.Minimize(objective), constraints)

        # 4. Extract the final optimization variables.
        self.P_dev = self.P_dev_vars[0]
        self.V_bus_ang = self.V_bus_ang_vars[0]

        return problem

    def _update_parameters(self, env):

        # Extract the fixed time series of demand and maximum generation.
        t_start = int(env.state[-1]) + 1   # time of day of next time step
        t_end = t_start + self.planning_steps
        P_loads = env.P_loads
        P_gen_pot = env.P_maxs

        # Deal with the case where the planning horizon spans over more than a
        # single day.
        while t_end > P_loads.shape[1]:
            P_loads = np.concatenate((P_loads, env.P_loads), axis=-1)
            P_gen_pot = np.concatenate((P_gen_pot, env.P_maxs), axis=-1)

        # Extract the P_loads for the next `planning_steps` steps.
        self.P_load_prev.value = P_loads[:, t_start: t_end] / self.baseMVA

        # Extract the potential generation P_pot from non-slack generators.
        self.P_gen_pot_prev.value = P_gen_pot[:, t_start: t_end] / self.baseMVA

        # Set the previous state of charge.
        full_state = env.simulator.state
        soc = [full_state['des_soc']['pu'][i] for i in self.des_ids]
        self.soc_prev.value = soc
