import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import datetime as dt
import time
import ast
import os
from importlib.machinery import SourceFileLoader

import gym_smartgrid.utils
from gym_smartgrid.simulator import Simulator
from gym_smartgrid.rendering import rendering
from gym_smartgrid.envs.utils import write_html
from gym_smartgrid import RENDERING_FOLDER, ENV_FILES


class SmartGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, folder, obs_values, delta_t=15, seed=None):

        # Load case.
        path_to_case = os.path.join(folder, ENV_FILES['case'])
        self.case = SourceFileLoader("case", path_to_case).load_module().network

        # Store paths to files needed for rendering.
        rel_path = os.path.relpath(folder, RENDERING_FOLDER)
        self.svg_data = {'network': os.path.join(rel_path, ENV_FILES['network'])}
        self.svg_data['labels'] = os.path.join(rel_path, ENV_FILES['svgLabels'])

        # Set random seed.
        self.seed(seed)

        self.delta_t = delta_t
        self.timestep_length = dt.timedelta(minutes=delta_t)
        self.episode_max_length = dt.timedelta(days=3 * 365)
        self.year = 2019

        self.obs_values = obs_values

        # Initialize AC power grid simulator.
        self.simulator = Simulator(self.case, delta_t=self.delta_t,
                                   rng=self.np_random)

        self.network_specs = self.simulator.network_specs
        self.action_space = self._build_action_space()
        self.observation_space = self._build_obs_space()

        dev_specs = self._get_dev_specs()
        self.generators = self.init_vre(dev_specs[2], dev_specs[3],
                                        self.timestep_length, self.np_random)

        self.loads = self.init_load(dev_specs[0], self.timestep_length,
                                    self.np_random)

    def _build_action_space(self):
        P_curt_bounds, alpha_bounds, q_bounds = self.simulator.get_action_space()

        space_curtailment = spaces.Box(low=P_curt_bounds[:, 1],
                                       high=P_curt_bounds[:, 0],
                                       dtype=np.float32)

        space_alpha = spaces.Box(low=alpha_bounds[:, 1],
                                 high=alpha_bounds[:, 0],
                                 dtype=np.float32)

        space_q = spaces.Box(low=q_bounds[:, 1], high=q_bounds[:, 0],
                             dtype=np.float32)

        return spaces.Tuple((space_curtailment, space_alpha, space_q))

    def _build_obs_space(self):
        obs_space = []
        network_specs = {k: np.array(v) for k, v in self.network_specs.items()}
        if 'P_BUS' in self.obs_values:
            space = spaces.Box(low=network_specs['PMIN_BUS'],
                               high=network_specs['PMAX_BUS'],
                               dtype=np.float32)
            obs_space.append(space)

        if 'Q_BUS' in self.obs_values:
            space = spaces.Box(low=network_specs['QMIN_BUS'],
                               high=network_specs['QMAX_BUS'],
                               dtype=np.float32)
            obs_space.append(space)

        if 'V_BUS' in self.obs_values:
            space = spaces.Box(low=network_specs['VMIN_BUS'],
                               high=network_specs['VMAX_BUS'],
                               dtype=np.float32)
            obs_space.append(space)

        if 'P_DEV' in self.obs_values:
            space = spaces.Box(low=network_specs['PMIN_DEV'],
                               high=network_specs['PMAX_DEV'],
                               dtype=np.float32)
            obs_space.append(space)

        if 'Q_DEV' in self.obs_values:
            space = spaces.Box(low=network_specs['QMIN_DEV'],
                               high=network_specs['QMAX_DEV'],
                               dtype=np.float32)
            obs_space.append(space)

        if 'I_BR' in self.obs_values:
            shape = network_specs['IMAX_BR'].shape
            space = spaces.Box(low=np.zeros(shape=shape),
                               high=network_specs['IMAX_BR'],
                               dtype=np.float32)
            obs_space.append(space)

        if 'SOC' in self.obs_values:
            space = spaces.Box(low=network_specs['SOC_MIN'],
                               high=network_specs['SOC_MAX'],
                               dtype=np.float32)
            obs_space.append(space)

        return spaces.Tuple(tuple(obs_space))

    def _get_dev_specs(self):
        load, power_plant, wind, solar = {}, {}, {}, {}
        for idx, dev_type in enumerate(self.network_specs['DEV_TYPE']):
            p_max = self.network_specs['PMAX_DEV'][idx]
            p_min = self.network_specs['PMIN_DEV'][idx]
            if dev_type == -1:
                load[idx] = p_min
            elif dev_type == 1:
                power_plant[idx] = p_max
            elif dev_type == 2:
                wind[idx] = p_max
            elif dev_type == 3:
                solar[idx] = p_max

        return load, power_plant, wind, solar

    def init_vre(self, wind_pmax, solar_pmax, delta_t, np_random):
        raise NotImplementedError

    def init_load(self, load_pmax, delta_t, np_random):
        raise NotImplementedError

    def step(self, action):

        # Check if the action is in the available action space.
        assert self.action_space.contains(action), "%r (%s) invalid" \
                                                   % (action, type(action))
        self._increment_t()

        # Get the output of the stochastic processes (vre generation, loads).
        P_loads = self.loads.next(self.time)
        self.P_gen_potential = self.generators.next(self.time)

        # Simulate a transition and compute the reward.
        reward = self.simulator.transition(P_loads, self.P_gen_potential, *action)
        self.state = self.simulator.state
        self.total_reward += reward

        # Create a tuple of observations.
        self.obs = self._get_observations()

        # End of episode if maximum number of timesteps has been reached.
        if self.time >= self.end_time:
            self.done = True

        # Information returned for debugging.
        info = None

        return self.obs, reward, self.done, info

    def _increment_t(self):
        self.time += self.timestep_length
        self.time = self.time.replace(year=self.year)

    def _get_observations(self):
        if self.state:
            obs = [list(self.state[ob]) for ob in self.obs_values]
        else:
            obs = None
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
        self.state = None
        self.render_mode = None
        self.simulator.reset()
        obs = self._get_observations()

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
            network_specs = self.simulator.compute_network_specs()
            self._init_render(network_specs)
        else:
            self._update_render(self.time, self._get_observations(),
                                list(self.P_gen_potential))

    def _init_render(self, network_specs):
        if self.render_mode in ['human', 'replay']:

            if self.svg_data is None:
                raise ValueError('svg data needs to be specified when '
                                 'initializing an instance of the environment '
                                 'for the rendering to be available.')
            write_html(self.svg_data)
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
        ns, obs, p_pot, times = self._unpack_history(history)

        self._init_render(ns)

        for i in range(len(obs)):
            self._update_render(times[i], obs[i], p_pot[i])

        self.close()

    def _unpack_history(self, history):
        ns = ast.literal_eval(history.network_specs[0])

        obs = history.obs[1:].values
        obs = [ast.literal_eval(o) for o in obs]

        p_potential = history.potential[1:].values
        p_potential = [ast.literal_eval(p) for p in p_potential]

        times = history.time[1:].values
        times = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in times]

        return ns, obs, p_potential, times

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
