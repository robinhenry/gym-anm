import os

from gym_smartgrid.envs import SmartGridEnv
from .scenarios import LoadGenerator, WindGenerator, SolarGenerator


class SmartGrid6(SmartGridEnv):
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


class SmartGrid6Easy(SmartGrid6):
    def __init__(self):
        super().__init__()

    def init_load(self, load_pmax, init_date, delta_t, np_random):
        scenario1 = [-1, -20, -0]
        scenario2 = [-5, -10, -20]
        scenario3 = [-1, -2, -0]

        loads = []
        for i in range(len(scenario1)):
            loads.append(self._constant_generator(scenario1[i], scenario2[i],
                                                  scenario3[i]))
        return loads

    def init_vre(self, wind_pmax, solar_pmax, init_date, delta_t, np_random):
        scenario1 = [30, 30]
        scenario2 = [5, 10]
        scenario3 = [0, 40]

        vres = []
        for i in range(len(scenario1)):
            vres.append(self._constant_generator(scenario1[i], scenario2[i],
                                                  scenario3[i]))
        return vres

    def init_soc(self, soc_max=None):
        return [s / 2. for s in soc_max]

    def _constant_generator(self, case1, case2, case3):
        transition_time = 8

        while True:

            # Scenario 3.
            if self.time.hour < 6:
                yield case3

            # Transition 3 -> 2.
            elif 6 <= self.time.hour < 8:
                for t in range(1, transition_time + 1):
                    diff = case3 - case2
                    yield (case3 - t * diff / transition_time)

            # Scenario 2.
            elif 8 <= self.time.hour < 11:
                yield case2

            # Transition 2 -> 1.
            elif 11 <= self.time.hour < 13:
                for t in range(1, transition_time + 1):
                    diff = case2 - case1
                    yield (case2 - t * diff / transition_time)

            # Scenario 1.
            elif 13 <= self.time.hour < 16:
                yield case1

            # Transition 1 -> 2.
            elif 16 <= self.time.hour < 18:
                for t in range(1, transition_time + 1):
                    diff = case1 - case2
                    yield (case1 - t * diff / transition_time)

            # Scenario 2.
            elif 18 <= self.time.hour < 21:
                yield case2

            # Transition 2 -> 3.
            elif 21 <= self.time.hour < 23:
                for t in range(1, transition_time + 1):
                    diff = case2 - case3
                    yield (case2 - t * diff / transition_time)

            # Scenario 3.
            else:
                yield case3


class SmartGrid6Hard(SmartGrid6):
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
