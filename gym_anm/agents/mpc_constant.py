import numpy as np

from .mpc import MPCAgent


class MPCAgentConstant(MPCAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forecast(self, env):

        # Extract the full state of the distribution network.
        full_state = env.simulator.state

        # Extract the current power injections from loads.
        P_load = [full_state['dev_p']['pu'][i] for i in self.load_ids]

        # Extract the current maximum generation from non-slack generators.
        P_pot = [full_state['gen_p_max']['pu'][i] for i in
                 self.non_slack_gen_ids]

        # Forecast constant values over the optimization horizon.
        P_load = np.array([P_load for _ in range(self.planning_steps)]).T
        P_pot = np.array([P_pot for _ in range(self.planning_steps)]).T

        return P_load, P_pot
