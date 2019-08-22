import os

from gym_smartgrid.envs import SmartGridEnv
from .scenarios import LoadGenerator, WindGenerator, SolarGenerator


class SmartGridEnv6(SmartGridEnv):
    def __init__(self):

        # Random seed.
        seed = 2019

        # Folder to new environment (== this folder).
        path_to_folder = os.path.dirname(os.path.realpath(__file__))

        # Values to include in the observation space.
        obs_values = ['P_DEV', 'Q_DEV', 'SOC']

        # Time interval between two time steps (minutes).
        delta_t = 15

        super().__init__(path_to_folder, obs_values, delta_t, seed)


class SmartGridEnv6Easy(SmartGridEnv6):
    def __init__(self):
        super().__init__()

    def init_load(self, load_pmax, init_date, delta_t, np_random):
        return [self._constant_generator(-5.),
                self._constant_generator(-10.),
                self._constant_generator(-20.)]

    def init_vre(self, wind_pmax, solar_pmax, init_date, delta_t, np_random):
        return [self._constant_generator(5.), self._constant_generator(10.)]

    def init_soc(self, soc_max=None):
        return soc_max

    def _constant_generator(self, value):
        while True:
            yield value


class SmartGridEnv6Hard(SmartGridEnv6):
    def __init__(self):
        super().__init__()

    def init_load(self, load_pmax, init_date, delta_t, np_random):
        folder = os.path.join(self.__file__, 'scenarios')
        loads = []
        for _, p_max in sorted(load_pmax.items()):
            new_load = LoadGenerator(folder, init_date, delta_t, np_random,
                                     p_max)
            loads.append(new_load)

        return loads

    def init_vre(self, wind_pmax, solar_pmax, init_date, delta_t, np_random):
        vres = [None] * (len(wind_pmax) + len(solar_pmax))

        for dev_id, p_max in sorted(wind_pmax.items()):
            vres[dev_id] = WindGenerator(init_date, delta_t, np_random, p_max)

        for dev_id, p_max in sorted(solar_pmax.items()):
            vres[dev_id] = SolarGenerator(init_date, delta_t, np_random, p_max)

        return vres

    def init_soc(self, soc_max=None):
        return None
