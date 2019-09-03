import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import datetime as dt

from gym_anm.simulator import Simulator


class ANMEnv(gym.Env):
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

    def __init__(self, network, obs_values, delta_t=15, seed=None):
        """
        Parameters
        ----------
        obs_values : list of str
            The values to include in the observation space.
        delta_t : int, optional
            The time interval between two consecutive time steps (minutes).
        seed : int, optional
            A random seed.
        """

        # Set random seed.
        self.seed(seed)

        # Time variables.
        self.delta_t = delta_t
        self.timestep_length = dt.timedelta(minutes=delta_t)
        self.year = 2019

        self.obs_values = obs_values

        # Initialize AC power grid simulator.
        self.simulator = Simulator(network, delta_t=self.delta_t)
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

    def init_dg(self, wind_pmax, solar_pmax, init_date, delta_t, np_random):
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
        self.generators = self.init_dg(dev_specs[2], dev_specs[3],
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

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError