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
from gym_smartgrid.rendering.py import rendering
from gym_smartgrid.envs.utils import write_html, sample_action
from gym_smartgrid import RENDERING_FOLDER, ENV_FILES
from gym_smartgrid.constants import RENDERED_STATE_VALUES, RENDERED_NETWORK_SPECS


class SmartGridEnv(gym.Env):
    """
    An environment simulating an electricity distribution network.

    This environment was designed to train Reinforcement Learning agents to
    perform well in Active Network Management (ANM) tasks in electricity
    distribution networks, where Variable Renewable Energy (VRE) curtailment is
    possible and distributed storage available.

    Attributes
    ----------
    case : dict of {str : numpy.ndarray}
        The input case file representing the electricity network.
    svg_data : dict of {str : str}
        A dictionary with keys {'network', 'labels'} and values storing the
        paths to the corresponding files needed for the environment rendering.
    delta_t : int
        The time interval between two consecutive time steps (minutes).
    timestep_length : datetime.timedelta
        The equivalent of `time_factor`.
    year : int
        The year on which to base the time process.
    obs_values : list of str
        The values to include in the observation space.
    simulator : Simulator
        The electricity distribution network simulator.
    network_specs : dict of {str : array_like}
        The operating characteristics of the electricity network.
    action_space : gym.spaces.Tuple
        The action space available to the agent interacting with the environment.
    observation_space : gym.spaces.Tuple
        The observation space available to the agent interacting with the
        environment.
    state : dict of {str : array_like}
        The current values of the state variables of the environment.
    total_reward : float
        The total reward accumulated so far.
    obs : list of list of float
        The current values of the state variables included in the observation
        space.
    time : datetime.datetime
        The current time.
    end_time : datetime.datetime
        The end time of the episode.
    init_soc : list of float
        The initial state of charge of each storage unit.
    done : bool
        True if the episode is over, False otherwise.
    render_mode : str
        The mode of the environment visualization.
    np_random : array_like
        The random seed.
    render_history : pandas.DataFrame
        The history of past states, used for later visualization.

    generators
    loads

    Methods
    -------
    init_vre()
    init_load()

    reset()
        Reset the environment.
    step(action)
        Take a control action and compute the associated reward.
    render(mode='human', sleep_time=0.1)
        Update the environment' state rendering.
    replay(path, sleep_time=0.1)
        Render a previously stored state history.
    close(path=None)
        Stop rendering.
    """

    metadata = {'render.modes': ['human', 'save']}

    def __init__(self, folder, obs_values, delta_t=15, seed=None):
        """
        Parameters
        ----------
        folder : str
            The absolute path to the folder providing the files to initialize a
            specific environment.
        obs_values : list of str
            The values to include in the observation space.
        delta_t : int, optional
            The time interval between two consecutive time steps (minutes).
        seed : int, optional
            A random seed.
        """

        # Load case.
        path_to_case = os.path.join(folder, ENV_FILES['case'])
        self.case = SourceFileLoader("case", path_to_case).load_module().network

        # Store paths to files needed for rendering.
        rel_path = os.path.relpath(folder, RENDERING_FOLDER)
        self.svg_data = {'network': os.path.join(rel_path, ENV_FILES['network'])}
        self.svg_data['labels'] = os.path.join(rel_path, ENV_FILES['svgLabels'])

        # Set random seed.
        self.seed(seed)

        # Time variables.
        self.delta_t = delta_t
        self.timestep_length = dt.timedelta(minutes=delta_t)
        self.year = 2019

        self.obs_values = obs_values

        # Initialize AC power grid simulator.
        self.simulator = Simulator(self.case, delta_t=self.delta_t)
        self.network_specs = self.simulator.specs

        # Build action and observation spaces.
        self.action_space = self._build_action_space()
        self.observation_space = self._build_obs_space()

    def _build_action_space(self):
        """
        Build the available action space.

        Returns
        -------
        gym.spaces.Tuple
            The action space of the environment.
        """

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
        """
        Build the observation space.

        Returns
        -------
        gym.spaces.Tuple
            The observation space.
        """

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

        if 'RATE' in self.obs_values:
            shape = network_specs['RATE'].shape
            space = spaces.Box(low=np.zeros(shape=shape),
                               high=network_specs['RATE'],
                               dtype=np.float32)
            obs_space.append(space)

        if 'SOC' in self.obs_values:
            space = spaces.Box(low=network_specs['SOC_MIN'],
                               high=network_specs['SOC_MAX'],
                               dtype=np.float32)
            obs_space.append(space)

        return spaces.Tuple(tuple(obs_space))

    def _get_dev_specs(self):
        """
        Extract the operating constraints of loads and VRE devices.

        Returns
        -------
        load, power_plant, solar : dict of {int : float}
            A dictionary of {key : value}, where the key is the device unique ID
            and the value the maximum real power injection of the corresponding
            device (negative for loads).
        """

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

    def init_vre(self, wind_pmax, solar_pmax, init_date, delta_t, np_random):
        """

        Parameters
        ----------
        wind_pmax
        solar_pmax
        delta_t
        np_random

        Returns
        -------

        """
        raise NotImplementedError('The function init_vre() should be implemented'
                                  ' by the subclass.')

    def init_load(self, load_pmax, init_date, delta_t, np_random):
        """

        Parameters
        ----------
        load_pmax
        delta_t
        np_random

        Returns
        -------

        """
        raise NotImplementedError('The function init_load() should be implemented'
                                  ' by the subclass.')

    def step(self, action):
        """
        Take a control action and transition from a state s_t to a state s_{t+1}.

        Parameters
        ----------
        action : Tuple of array_like
            The action taken by the agent.

        Returns
        -------
        state_values : Tuple of array_like
            The observation corresponding to the new state s_{t+1}.
        reward : float
            The rewar associated with the transition.
        done : bool
            True if the episode is over, False otherwise.
        info : dict
            A dictionary of further information.
        """

        if self.end_time is not None and self.time >= self.end_time:
            raise gym.error.ResetNeeded('The episode is already over.')

        # Check if the action is in the available action space.
        assert self.action_space.contains(action), "%r (%s) invalid" \
                                                   % (action, type(action))

        # Get the output of the stochastic processes (vre generation, loads).
        P_loads = [next(load) for load in self.loads]
        self.P_gen_potential = [next(gen) for gen in self.generators]

        # Simulate a transition and compute the reward.
        self.state, reward, self.e_loss, self.penalty = \
            self.simulator.transition(P_loads, self.P_gen_potential, *action)
        self.total_reward += reward

        # Create a tuple of observations.
        self.obs = self._get_observations()

        # End of episode if maximum number of time steps has been reached.
        self._increment_t()
        if self.end_time is not None and self.time >= self.end_time:
            self.done = True

        # Information returned for debugging.
        info = None

        return self.obs, reward, self.done, info

    def _increment_t(self):
        """ Increment the time. """
        self.time += self.timestep_length

        if self.time.year != self.year:
            self.year_counter += 1
            self.time = self.time.replace(year=self.year)

    def _get_observations(self):
        """
        Select the observations available to the agent from the current state.

        Returns
        -------
        state_values : list of list of float
            The observations available to the agent, as specified by
            `self.obs_values`.
        """

        if self.state:
            obs = [list(self.state[ob]) for ob in self.obs_values]
        else:
            obs = None
        return obs

    def seed(self, seed=None):
        """ Seed the random number generator. """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset the environment.

        Returns
        -------
        state_values : list of list of float
            The observations available to the agent, as specified by
            `self.obs_values`.
        """

        self.done = False
        self.render_mode = None
        self.total_reward = 0.
        self.state = None

        # Always start the simulation on January, 1st.
        self.time = dt.datetime(self.year, 1, 1)
        self.end_time = None
        self.year_counter = 0

        # Initialize stochastic processes.
        dev_specs = self._get_dev_specs()
        self.generators = self.init_vre(dev_specs[2], dev_specs[3],
                                        self.time, self.timestep_length,
                                        self.np_random)
        self.loads = self.init_load(dev_specs[0], self.time,
                                    self.timestep_length,
                                    self.np_random)

        # Reset the initial SoC of each storage unit.
        soc_start = self.init_soc(self.network_specs['SOC_MAX'])
        self.simulator.reset(soc_start)

        # Initialize simulator with an action that does nothing.
        self.step(self._init_action())
        self.time -= self.timestep_length

        # self.step(sample_action(self.np_random, self.action_space))
        self.total_reward = 0.

        # Get the initial observations.
        self.state = self.simulator.state
        obs = self._get_observations()

        return obs

    def _init_action(self):
        """
        Get the action that doesn't modify the state of the system.

        Returns
        -------
        action : tuple of numpy.ndarray
            The action to take which will not modify the state.

        """

        curt = self.action_space.spaces[0].high
        action = (curt, np.array([0.]), np.array([0.]))

        return action

    def init_soc(self, soc_max=None):
        """
        Get the initial state of charge for each storage unit.

        Parameters
        ----------
        soc_max : list of float
            The maximum state of charge of each storage unit.

        Returns
        -------
        list of float
            The initial state of charge of each storage unit.
        """

        raise NotImplementedError('The function init_soc() should be implemented'
                                  ' by the subclass.')

    def render(self, mode='human', sleep_time=0.1):
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : {'human', 'save'}, optional
            The mode of rendering. If 'human', the environment is rendered while
            the agent interacts with it. If 'save', the state history is saved
            for later visualization.
        sleep_time : float, optional
            The sleeping time between two visualization updates.

        Raises
        ------
        NotImplementedError
            If a non-valid mode is specified.

        See Also
        --------
        replay()
        """

        if self.render_mode is None:

            if mode in ['human', 'replay']:
                pass
            elif mode == 'save':
                self.render_history = None
            else:
                raise NotImplementedError

            self.render_mode = mode
            specs = [list(self.network_specs[s]) for s in RENDERED_NETWORK_SPECS]
            self._init_render(specs)

            # Render the initial state.
            self.render(mode=mode, sleep_time=1.)

        else:
            state_values = [list(self.state[s]) for s in RENDERED_STATE_VALUES]
            self._update_render(self.time - self.timestep_length,
                                state_values,
                                list(self.P_gen_potential),
                                [self.e_loss, self.penalty],
                                sleep_time=sleep_time)

    def _init_render(self, network_specs):
        """
        Initialize the rendering of the environment state.

        Parameters
        ----------
        network_specs : dict of {str : list}
            The operating characteristics of the electricity distribution network.

        Raises
        ------
        NotImplementedError
            If the rendering mode is non-valid.
        """

        title = type(self).__name__

        if self.render_mode in ['human', 'replay']:
            write_html(self.svg_data)
            self.http_server, self.ws_server = rendering.start(
                title,
                *network_specs)

        elif self.render_mode == 'save':
            s = pd.Series({'title': title, 'specs': network_specs})
            self.render_history = pd.DataFrame([s])

        else:
            raise NotImplementedError

    def _update_render(self, cur_time, state_values, P_potential, costs,
                       sleep_time):
        """
        Update the rendering of the environment state.

        Parameters
        ----------
        cur_time : datetime.datetime
            The time corresponding to the current time step.
        state_values : list of list of float
            The state values needed for rendering.
        P_potential : list of float
            The potential generation of each VRE before curtailment (MW).
        costs : list of float
            The total energy loss and the total penalty associated with operating
            constraints violation.
        sleep_time : float
            The sleeping time between two visualization updates.

        Raises
        ------
        NotImplementedError
            If the rendering mode is non-valid.
        """



        if self.render_mode in ['human', 'replay']:
            rendering.update(self.ws_server.address,
                             cur_time,
                             *state_values,
                             P_potential,
                             costs)
            time.sleep(sleep_time)

        elif self.render_mode == 'save':
            d = {'time': self.time,
                 'state_values': state_values,
                 'potential': P_potential,
                 'costs': costs}
            s = pd.Series(d)
            self.render_history = self.render_history.append(s,
                                                             ignore_index=True)

        else:
            raise NotImplementedError

    def replay(self, path, sleep_time=0.1):
        """
        Render a state history previously saved.

        Parameters
        ----------
        path : str
            The path to the saved history.
        sleep_time : float, optional
            The sleeping time between two visualization updates.
        """

        self.reset()
        self.render_mode = 'replay'

        history = pd.read_csv(path)
        ns, obs, p_pot, times, costs = self._unpack_history(history)

        self._init_render(ns)

        for i in range(len(obs)):
            self._update_render(times[i], obs[i], p_pot[i], costs, sleep_time)

        self.close()

    def _unpack_history(self, history):
        """
        Unpack a previously stored history of state variables.

        Parameters
        ----------
        history : pandas.DataFrame
            The history of states, with fields {'specs', 'time', 'state_values',
            'potential'}.

        Returns
        -------
        ns : dict of {str : list}
            The operating characteristics of the electricity distribution network.
        state_values : list of list of float
            The state values needed for rendering.
        p_potential : list of float
            The potential generation of each VRE before curtailment (MW).
        times : list of datetime.datetime
            The times corresponding to each time step.
        costs : list of float
            The total energy loss and the total penalty associated with operating
            constraints violation.
        """

        ns = ast.literal_eval(history.specs[0])

        state_values = history.state_values[1:].values
        state_values = [ast.literal_eval(o) for o in state_values]

        p_potential = history.potential[1:].values
        p_potential = [ast.literal_eval(p) for p in p_potential]

        times = history.time[1:].values
        times = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in times]

        costs = history.costs[1:].values
        costs = [ast.literal_eval(c) for c in costs]

        return ns, state_values, p_potential, times, costs

    def close(self, path=None):
        """
        Close the rendering.

        Parameters
        ----------
        path : str, optional
            The path to the file to store the state history, only used if
            `render_mode` is 'save'.

        Returns
        -------
        pandas.DataFrame
            The state history.
        """

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
