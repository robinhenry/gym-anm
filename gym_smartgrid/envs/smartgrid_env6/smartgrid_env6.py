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
        folder_house = os.path.join(os.path.dirname(__file__), 'scenarios',
                                    'data_demand_curves', 'house')
        folder_factory = os.path.join(os.path.dirname(__file__), 'scenarios',
                                    'data_demand_curves', 'factory')
        loads = []

        for dev_idx, p_max in sorted(load_pmax.items()):

            # Assign the folder corresponding to the kind of load.
            if dev_idx in [1, 5]:
                folder = folder_house
            elif dev_idx in [3]:
                folder = folder_factory
            else:
                raise ValueError('The device ID does not match.')

            new_load = LoadGenerator(folder, init_date, delta_t, np_random,
                                     p_max)
            loads.append(new_load)

        return loads

    def init_vre(self, wind_pmax, solar_pmax, init_date, delta_t, np_random):
        vres = {}

        for dev_id, p_max in sorted(wind_pmax.items()):
            vres[dev_id] = WindGenerator(init_date, delta_t, np_random, p_max)

        for dev_id, p_max in sorted(solar_pmax.items()):
            vres[dev_id] = SolarGenerator(init_date, delta_t, np_random, p_max)

        # Transform the dictionary into a list, ordered by device index.
        vres_lst = []
        for dev_id in sorted(vres.keys()):
            vres_lst.append(vres[dev_id])

        return vres_lst

    def init_soc(self, soc_max=None):
        return None
