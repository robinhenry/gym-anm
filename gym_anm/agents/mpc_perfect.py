"""An MPC-based policy with perfect forecasts."""
import numpy as np

from .mpc import MPCAgent


class MPCAgentPerfect(MPCAgent):
    """
    A Model Predictive Control (MPC)-based policy with perfect forecasts.

    This class implements the :math:`\\pi_{MPC-N}^{perfect}` policy, a variant
    of the general :math:`\\pi_{MPC-N}` policy in which the future demand and
    generation are perfectly predicted (i.e., assumed known).

    For more information, see https://gym-anm.readthedocs.io/en/latest/topics/mpc.html#perfect-forecast.
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
