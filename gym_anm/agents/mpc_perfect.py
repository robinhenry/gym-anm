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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forecast(self, env):

        # Extract the fixed time series of demand and maximum generation.
        t_start = int(env.state[-1]) + 1  # time of day of next time step
        t_end = t_start + self.planning_steps
        P_loads = env.P_loads
        P_gen_pot = env.P_maxs

        # Deal with the case where the planning horizon spans over more than a
        # single day.
        while t_end > P_loads.shape[1]:
            P_loads = np.concatenate((P_loads, env.P_loads), axis=-1)
            P_gen_pot = np.concatenate((P_gen_pot, env.P_maxs), axis=-1)

        # Extract the P_loads for the next `planning_steps` steps.
        P_load_forecast = P_loads[:, t_start: t_end] / self.baseMVA

        # Extract the potential generation P_pot from non-slack generators.
        P_gen_forecast = P_gen_pot[:, t_start: t_end] / self.baseMVA

        return P_load_forecast, P_gen_forecast
