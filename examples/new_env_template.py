"""
This file gives the template to follow when creating new gym-anm environments.

For more information, see https://gym-anm.readthedocs.io/en/latest/topics/design_new_env.html.
"""
from gym_anm import ANMEnv

class CustomEnvironment(ANMEnv):

    def __init__(self):
        network = {'baseMVA': ...,
                   'bus': ...,
                   'device': ...,
                   'branch': ...}  # power grid specs
        observation = ...          # observation space
        K = ...                    # number of auxiliary variables
        delta_t = ...              # time interval between timesteps
        gamma = ...                # discount factor
        lamb = ...                 # penalty weighting hyperparameter
        aux_bounds = ...           # bounds on auxiliary variable (optional)
        costs_clipping = ...       # reward clipping parameters (optional)
        seed = ...                 # random seed (optional)

        super().__init__(network, observation, K, delta_t, gamma, lamb,
                         aux_bounds, costs_clipping, seed)

    def init_state(self):
        ...

    def next_vars(self, s_t):
        ...

    def observation_bounds(self):  # optional
        ...

    def render(self, mode='human'):  # optional
        ...

    def close(self):  # optional
        ...