import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import datetime
import time
import ast

import gym_smartgrid.utils
from gym_smartgrid.smartgrid_simulator import Simulator
from gym_smartgrid.smartgrid_simulator import utils
from gym_smartgrid.visualization import rendering


class SmartGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, case):

        # Set random seed.
        self.seed()

        self.timestep_length = datetime.timedelta(minutes=15)
        self.episode_max_length = datetime.timedelta(days=3 * 365)
        self.year = 2019

        # Initialize AC power grid simulator.
        self.simulator = Simulator(case, self.np_random)

        # Initialize action space for each action type.
        P_curt_bounds, alpha_bounds, q_bounds = self.simulator.get_action_space()

        space_curtailment = spaces.Box(low=P_curt_bounds[1, :],
                                       high=P_curt_bounds[0, :],
                                       dtype=np.float32)

        space_alpha = spaces.Box(low=alpha_bounds[1, :],
                                 high=alpha_bounds[0, :],
                                 dtype=np.float32)

        space_q = spaces.Box(low=q_bounds[1, :], high=q_bounds[0, :],
                             dtype=np.float32)

        # Initialize the global action space to be the product of 3 subspaces.
        self.action_space = spaces.Tuple((space_curtailment, space_alpha,
                                          space_q))

        # Initialize observation space made of:
        # - P, Q injection at each bus (2 * N),
        # - I magnitude in each transmission line,
        # - SoC at each storage unit (N_storage).
        p_obs = spaces.Box(low=-np.inf, high=np.inf,
                           shape=(self.simulator.N_bus,),
                           dtype=np.float32)
        q_obs = p_obs

        i_obs = spaces.Box(low=0., high=np.inf, shape=(self.simulator.N_branch,),
                           dtype=np.float32)

        soc_obs = spaces.Box(low=np.zeros(shape=(self.simulator.N_storage,)),
                             high=self.simulator.max_soc, dtype=np.float32)

        self.observation_space = spaces.Tuple((p_obs, q_obs, i_obs, soc_obs))

        # Initialize distributed generators (stochastic processes).
        wind_pmax, solar_pmax, load_pmax = self.simulator.get_vre_specs()

        self.generators = utils.init_vre(wind_pmax, solar_pmax,
                                         self.timestep_length,
                                         np_random=self.np_random)

        # Initialize load stochastic processes.
        self.loads = utils.init_load(load_pmax, self.timestep_length,
                                self.np_random)

    def step(self, action):

        # Check if the action is in the available action space.
        assert self.action_space.contains(action), "%r (%s) invalid" \
                                                   % (action, type(action))

        self._increment_t()

        # Get the output of the stochastic processes (vre generation, loads).
        P_loads = self.loads.next(self.time)
        P_gen_potential = self.generators.next(self.time)
        self.P_gen_potential = P_gen_potential

        # Simulate a transition and compute the reward.
        reward = self.simulator.transition(P_loads, P_gen_potential, *action)
        self.total_reward += reward

        # Create a (4,) tuple of observations.
        obs = self._get_obs()

        # End of episode if maximum number of timesteps has been reached.
        if self.time >= self.end_time:
            self.done = True

        # Information returned for debugging.
        info = None

        return obs, reward, self.done, info

    def _increment_t(self):
        self.time += self.timestep_length
        self.time = self.time.replace(year=self.year)

    def _get_obs(self):
        # Create a (4,) tuple of observations.
        obs = list(self.simulator.P_device), \
              list(self.simulator.Q_device), \
              list(self.simulator.I_br_magn), \
              list(self.simulator.SoC), \
              list(self.simulator.P_br)
        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Select random date.
        self.time = gym_smartgrid.utils.random_date(self.np_random, self.year)
        self.end_time = self.time + self.episode_max_length

        self.total_reward = 0.
        self.done = False
        self.render_mode = None
        self.simulator.reset()
        obs = self._get_obs()

        return obs

    def render(self, mode='human', sleep_time=0.1):
        if self.render_mode is None:
            if mode == 'human':
                self.sleep_time = sleep_time
            elif mode == 'save':
                self.render_history = None
            else:
                raise NotImplementedError
            self.render_mode = mode
            network_specs = self.simulator.get_network_specs()
            self._init_render(network_specs)
        else:
            self._update_render(self.time, self._get_obs(),
                                list(self.P_gen_potential))

    def _init_render(self, network_specs):
        if self.render_mode in ['human', 'replay']:
            self.http_server, self.ws_server = rendering.start(
                *network_specs)
        elif self.render_mode == 'save':
            s = pd.Series({'network_specs': network_specs})
            self.render_history = pd.DataFrame([s])
        else:
            raise NotImplementedError

    def _update_render(self, cur_time, obs, P_potential):
        if self.render_mode in ['human', 'replay']:
            rendering.update(self.ws_server.address,
                             cur_time,
                             obs[0],
                             obs[2],
                             obs[3],
                             obs[4],
                             P_potential)
            time.sleep(self.sleep_time)

        elif self.render_mode == 'save':
            d = {'time': self.time,
                 'obs': obs,
                 'potential': P_potential}
            s = pd.Series(d)
            self.render_history = self.render_history.append(s,
                                                             ignore_index=True)
        else:
            raise NotImplementedError

    def replay(self, path, sleep_time=0.1):
        self.reset()

        self.render_mode = 'replay'
        self.sleep_time = sleep_time

        history = pd.read_csv(path)
        network_specs = ast.literal_eval(history.network_specs[0])

        obs = ast.literal_eval(history.obs[1:])
        p_potential = ast.literal_eval(history.potential[1:])
        times = ast.literal_eval(history.time[1:])

        self._init_render(network_specs)

        for i in range(obs.size):
            self._update_render(times[i], obs[i], p_potential[i])

        self.close()

    def close(self, path=None):
        to_return = None
        if self.render_mode in ['human', 'replay']:
            rendering.close(self.http_server, self.ws_server)
        if self.render_mode == 'save':
            if path is None:
                raise ValueError('No path specified to save the history.')
            self.render_history.to_csv(path, index = None, header=True)
            to_return = self.render_history

        self.render_mode = None

        return to_return
