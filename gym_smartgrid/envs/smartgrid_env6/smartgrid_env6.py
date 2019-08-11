import os
from gym_smartgrid.envs import SmartGridEnv
from .utils import init_load as i_load
from .utils import init_vre as i_vre

class SmartGridEnv6(SmartGridEnv):
    def __init__(self):

        seed = 2019


        # Folder to new environment (== this folder).
        path_to_folder = os.path.dirname(os.path.realpath(__file__))
        obs_values = ['P_BUS', 'Q_BUS', 'I_BR', 'SOC']
        delta_t = 15

        super().__init__(path_to_folder, obs_values, delta_t, seed)

    def init_load(self, load_pmax, delta_t, np_random):
        return i_load(load_pmax, delta_t, np_random)

    def init_vre(self, wind_pmax, solar_pmax, delta_t, np_random):
        return i_vre(wind_pmax, solar_pmax, delta_t, np_random=np_random)

