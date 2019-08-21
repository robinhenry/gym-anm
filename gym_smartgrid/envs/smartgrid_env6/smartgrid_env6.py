import os
from gym_smartgrid.envs import SmartGridEnv


class SmartGridEnv6Easy(SmartGridEnv):
    def __init__(self):

        # Random seed.
        seed = 2019

        # Folder to new environment (== this folder).
        path_to_folder = os.path.dirname(os.path.realpath(__file__))

        # Values to include in the observation space.
        obs_values = ['P_DEV', 'Q_DEV', 'SOC']

        # Time interval between two time steps.
        delta_t = 15

        super().__init__(path_to_folder, obs_values, delta_t, seed)

    def init_load(self, load_pmax, delta_t, np_random):
        return [_constant_generator(-5.),
                _constant_generator(-10.),
                _constant_generator(-20.)]

    def init_vre(self, wind_pmax, solar_pmax, delta_t, np_random):
        return [_constant_generator(5.), _constant_generator(10.)]

    def init_soc(self, soc_max=None):
        return soc_max


def _constant_generator(value):
    while True:
        yield value

