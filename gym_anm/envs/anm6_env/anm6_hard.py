import os

from gym_anm.envs.anm6_env.anm6 import ANM6
from gym_anm.envs.anm6_env.scenarios import LoadGenerator, WindGenerator,\
    SolarGenerator


class ANM6Hard(ANM6):
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

    def init_soc(self, soc_max):
        return None