import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import datetime
import time

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

        self.total_reward = 0.
        self.done = False
        self.render_mode = None
        self.render_speed = {0: 0, 1: 0.1, 2: 0.5, 3: 2}

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
        obs = np.array(self.simulator.P_device), \
              np.array(self.simulator.Q_device), \
              np.array(self.simulator.I_br_magn), \
              np.array(self.simulator.SoC), \
              np.array(self.simulator.P_br)
        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Select random date.
        self.time = gym_smartgrid.utils.random_date(self.np_random, self.year)
        self.end_time = self.time + self.episode_max_length

        self.total_reward = 0.
        self.simulator.reset()
        obs = self._get_obs()

        return obs

    def render(self, mode='human', speed=0):
        if mode == 'human':
            if self.render_mode is None:
                self.render_mode = 'human'
                self.speed_level = speed
                self._render_init()
            else:
                self._render_update()
        elif mode == 'save':
            if self.render_mode is None:
                self.render_mode = 'save'
                self._save_init()
            else:
                self._save_update()
        else:
            raise NotImplementedError
        
    def _save_init(self):
        pass

    def _save_update(self):
        pass

    def _render_init(self):
        network_specs = self.simulator.get_network_specs()
        self.http_server, self.ws_server = rendering.start(
            *network_specs)

    def _render_update(self):
        obs = self._get_obs()
        rendering.update(self.ws_server.address,
                         self.time,
                         list(obs[0]),
                         list(obs[2]),
                         list(obs[3]),
                         list(obs[4]),
                         list(self.P_gen_potential))

        time.sleep(self.render_speed[self.speed_level])

    def close(self):
        rendering.close(self.http_server, self.ws_server)
        self.render_mode = None
