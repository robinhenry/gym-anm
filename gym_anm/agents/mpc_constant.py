"""An MPC-based policy with constant forecasts."""
import numpy as np

from .mpc import MPCAgent


class MPCAgentConstant(MPCAgent):
    """
    A Model Predictive Control (MPC)-based policy with constant forecasts.

    This class implements the :math:`\\pi_{MPC-N}^{constant}` policy, a variant
    of the general :math:`\\pi_{MPC-N}` policy in which the future demand and
    generation are assumed constant over the optimization horizon.

    For more information, see https://gym-anm.readthedocs.io/en/latest/topics/mpc.html#constant-forecast.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forecast(self, env):
        # Extract the full state of the distribution network.
        full_state = env.simulator.state

        # Extract the current power injections from loads.
        P_load_forecast = [full_state['dev_p']['pu'][i] for i in self.load_ids]

        # Extract the current maximum generation from non-slack generators.
        P_gen_forecast = [full_state['gen_p_max']['pu'][i] for i in
                          self.non_slack_gen_ids]

        # Forecast constant values over the optimization horizon.
        P_load_forecast = np.array([P_load_forecast for _ in range(self.planning_steps)]).T
        P_gen_forecast = np.array([P_gen_forecast for _ in range(self.planning_steps)]).T

        return P_load_forecast, P_gen_forecast
