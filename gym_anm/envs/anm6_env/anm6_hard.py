import os

from gym_anm.envs.anm6_env.anm6 import ANM6
from gym_anm.envs.anm6_env.scenarios import LoadGenerator, WindGenerator,\
    SolarGenerator


class ANM6Hard(ANM6):
    def __init__(self):
        super().__init__()

    def init_dg_load(self, pmax, init_date, delta_t, np_random):

        folder_house_ev = os.path.join(os.path.dirname(__file__), 'scenarios',
                                    'data_demand_curves', 'house')
        folder_factory = os.path.join(os.path.dirname(__file__), 'scenarios',
                                    'data_demand_curves', 'factory')

        iterators = []
        for idx, dev in enumerate(pmax):
            dev_type, p_max = dev

            if dev_type == -1:

                # Assign the folder corresponding to the kind of load.
                if idx in [0, 4]:  # shifted by 1 because slack bus is not here.
                    folder = folder_house_ev
                elif idx == 2:
                    folder = folder_factory
                else:
                    raise ValueError('The device ID does not match.')

                new_dev = LoadGenerator(folder, init_date, delta_t, np_random,
                                        p_max)

            elif dev_type == 2:
                new_dev = WindGenerator(init_date, delta_t, np_random, p_max)

            elif dev_type == 3:
                new_dev = SolarGenerator(init_date, delta_t, np_random, p_max)

            else:
                raise ValueError('Device type is not supported.')

            iterators.append(new_dev)

        return iterators

    def init_soc(self, soc_max):
        return None