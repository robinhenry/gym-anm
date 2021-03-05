"""
This file contains an example of a custom gym-anm environment that
inherits from ANM6.

Features:
* rendering is available (since ANM6 is inherited),
* it uses the same 6-bus 7-device power grid as the ANM6Easy-v0 task,
* the initial state s0 is randomly sampled (see `init_state()`),
* load demands and maximum generations are randomly sampled from
  within their physical limits (see `next_vars()`),
* the same auxiliary variable as in ANM6Easy-v0 is used, indicating
  the time of day as an index in [0, 96] (see `next_vars()`),

For more information, see https://gym-anm.readthedocs.io/en/latest/topics/rendering.html.
"""
import numpy as np
from gym_anm.envs import ANM6

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
        n_dev = self.simulator.N_device                # number of devices
        n_des = self.simulator.N_des                   # number of DES units
        n_gen = self.simulator.N_non_slack_gen         # number of non-slack generators
        s = np.random.rand(2 * n_dev + n_des + n_gen)  # random state vector

        # Let the auxiliary variable be a time of day index where increments
        # represent `self.delta_t` time durations.
        # Initial time: 00:00.
        aux = 0

        return np.hstack((s, aux))                 # initial state vector s0

    def next_vars(self, s_t):
        """ Generate the next stochastic variables and auxiliary variables."""
        next_var = []

        # Random demand for residential area in [-10, 0] MW.
        next_var.append(-10 * np.random.rand(1)[0])

        # Random PV max generation in [0, 30] MW.
        next_var.append(30 * np.random.rand(1)[0])

        # Random demand for industrial complex in [-30, 0] MW.
        next_var.append(-30 * np.random.rand(1)[0])

        # Random wind farm max generation in [0, 50] MW.
        next_var.append(50 * np.random.rand(1)[0])

        # Random load from EV charging station in [-30, 0] MW.
        next_var.append(-30 * np.random.rand(1)[0])

        # Auxiliary variable is the time of day index in [0, 96].
        aux = int((s_t[-1] + 1) % (24 / self.delta_t))
        next_var.append(aux)

        return np.array(next_var)

if __name__ == '__main__':
    env = CustomANM6Environment()
    env.reset()

    for t in range(10):
        a = env.action_space.sample()
        o, r, done, _ = env.step(a)
        print(f't={t}, r_t={r:.3}')