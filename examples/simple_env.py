"""
This file contains an example of a custom gym-anm environment.

Features:
* it uses a 2-bus power grid: Slack generator (bus 0) --- Load (bus 1),
* the initial state s0 is randomly generated (see `init_state()`),
* load demands are randomly generated in [-10, 0] (see `next_vars()`),
* a random auxiliary variable is added for illustrating the process of
  using them (it is useless in this case) (see `next_vars()`).

For more information, see https://gym-anm.readthedocs.io/en/latest/topics/design_new_env.html.
"""
import numpy as np
from gym_anm import ANMEnv

"""
A 2-bus power grid with topology:
    Slack (bus 0) ---- Load (bus 1)
"""
network = {
    'baseMVA': 100,
    'bus': np.array([[0, 0, 132, 1., 1.],
                     [1, 1, 33, 1.1, 0.9]]),
    'device': np.array([
        [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None],
        [1, 1, -1, 0.2, 0, -10,  None, None, None, None, None, None, None, None, None]]),
    'branch': np.array([[0, 1, 0.01, 0.1, 0., 3, 1, 0]])
}


class SimpleEnvironment(ANMEnv):
    """An example of a simple 2-bus custom gym-anm environment."""

    def __init__(self):
        observation = 'state'             # fully observable environment
        K = 1                             # 1 auxiliary variable
        delta_t = 0.25                    # 15min intervals
        gamma = 0.9                       # discount factor
        lamb = 100                        # penalty weighting hyperparameter
        aux_bounds = np.array([[0, 10]])  # bounds on auxiliary variable
        costs_clipping = (1, 100)         # reward clipping parameters
        seed = 1                          # random seed

        super().__init__(network, observation, K, delta_t, gamma, lamb,
                         aux_bounds, costs_clipping, seed)

    def init_state(self):
        """Return a state vector with random values in [0, 1]."""
        n_dev = self.simulator.N_device              # number of devices
        n_des = self.simulator.N_des                 # number of DES units
        n_gen = self.simulator.N_non_slack_gen       # number of non-slack generators
        N_vars = 2 * n_dev + n_des + n_gen + self.K  # size of state vectors

        return np.random.rand(N_vars)                # random state vector

    def next_vars(self, s_t):
        """Return a random load injection in [-10, 0] and a random aux variable in [0,10]."""
        P_load = -10 * np.random.rand(1)[0]        # Random demand in [-10, 0]
        aux = np.random.randint(0, 10)             # Random auxiliary variable in [0, 10]

        return np.array([P_load, aux])

if __name__ == '__main__':
    env = SimpleEnvironment()
    env.reset()

    for t in range(10):
        a = env.action_space.sample()
        o, r, done, _ = env.step(a)
        print(f't={t}, r_t={r:.3}')