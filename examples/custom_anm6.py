import numpy as np
from gym_anm.envs import ANM6

"""
Sub-classing ANM6 makes the rendering of the environment available.
"""


class CustomANM6Environment(ANM6):
    """A gym-anm task built on top of the ANM6 grid."""

    def __init__(self):
        observation = 'state'             # fully observable environment
        K = 1                             # 1 auxiliary variable
        delta_t = 0.25                    # 15min intervals
        gamma = 0.9                       # discount factor
        lamb = 100                        # penalty weighting hyperparameter
        aux_bounds = np.array([[0, 10]])  # bounds on auxiliary variable
        costs_clipping = (1, 100)         # reward clipping parameters
        seed = 1                          # random seed

        super().__init__(observation, K, delta_t, gamma, lamb,
                         aux_bounds, costs_clipping, seed)

    def init_state(self):
        """Return a state vector with random values in [0, 1]."""
        n_dev = self.simulator.N_device          # number of devices
        n_des = self.simulator.N_des             # number of DES units
        n_gen = self.simulator.N_non_slack_gen   # number of non-slack generators
        N_vars = n_dev + n_des + n_gen + self.K  # size of state vectors

        return np.random.rand(N_vars)            # random state vector

    def next_vars(self, s_t):
        """Return a random load injection in [-10, 0] and a random aux variable in [0,10]."""
        P_load = -10 * np.random.rand(1)[0]        # Random demand in [-10, 0]
        aux = np.random.randint(0, 10)             # Random auxiliary variable in [0, 10]

        return np.array([P_load, aux])


if __name__ == '__main__':
    env = CustomANM6Environment()
    env.reset()

    for t in range(10):
        a = env.action_space.sample()
        o, r, done, _ = env.step(a)
        print(f't={t}, r_t={r:.3}')