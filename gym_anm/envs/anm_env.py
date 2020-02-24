import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import datetime as dt

from gym_anm.simulator import Simulator
from gym_anm.envs import utils


class ANMEnv(gym.Env):
    """
    An gym environment simulating an electricity distribution network.

    This environment was designed to train Reinforcement Learning agents to
    perform well in Active Network Management (ANM) tasks in electricity
    distribution networks, where Variable Renewable Energy (VRE) curtailment is
    possible and distributed storage available.

    Attributes
    ----------
    action_space : gym.spaces.Tuple
        The action space available to the agent interacting with the environment.
    delta_t : int
        The time interval between two consecutive time steps (minutes).
    network_specs : dict of {str : list}
        The operating characteristics of the electricity network.
    np_random : numpy.random.RandomState
        The random state of the environment.
    obs_values : list of str
        The values to include in the observation space.
    observation_space : gym.spaces.Tuple
        The observation space available to the agent interacting with the
        environment.
    simulator : Simulator
        The electricity distribution network simulator.
    timestep_length : datetime.timedelta
        Equivalent to `delta_t`.
    year : int
        The year on which to base the time process.




    svg_data : dict of {str : str}
        A dictionary with keys {'network', 'labels'} and values storing the
        paths to the corresponding files needed for the environment rendering.




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
        network : dict of {str : numpy.ndarray}
            The network input file describing the power grid.
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

        # Set the observation space for the environment.
        utils.check_obs_values(obs_values)
        self.obs_values = obs_values

        # Initialize AC power grid simulator.
        self.simulator = Simulator(network, delta_t=self.delta_t)
        self.network_specs = self.simulator.specs

        # Build action and observation spaces.
        self.action_space, self.action_lengths, self.action_high, \
            self.action_low = self._build_action_space()
        self.observation_space, self.obs_low, self.obs_high, \
            self.obs_space_bounded = self._build_obs_space()

    def _build_action_space(self):
        """
        Build the available action space.

        Notes
        -----
        The returned action space is normalized so that the environment accepts
        actions in the range [-1, 1]. The conversion is then handled by the
        environment in `step()`.

        Returns
        -------
        space : gym.spaces.Box
            The normalized action space of the environment.
        action_lengths : list of int
            The number of action variables for each type of action, e.g.
            [4, 2, 2] -> 4 renewable generators, 2 DES units.
        lower_bounds : numpy.ndarray
            The actual lower bounds on actions accepted by the power grid
            simulator.
        upper_bounds : numpy.ndarray
            The actual upper bounds on actions accepted by the power grid
            simulator.
        """

        P_curt_bounds, alpha_bounds, q_bounds = self.simulator.get_action_space()

        lower_bounds = np.concatenate((P_curt_bounds[:, 1],
                                       alpha_bounds[:, 1],
                                       q_bounds[:, 1]))

        upper_bounds = np.concatenate((P_curt_bounds[:, 0],
                                       alpha_bounds[:, 0],
                                       q_bounds[:, 0]))

        space = spaces.Box(low=-np.ones_like(lower_bounds),
                           high=np.ones_like(upper_bounds))

        action_lengths = [P_curt_bounds.shape[0], alpha_bounds.shape[0],
                          q_bounds.shape[0]]

        return space, action_lengths, lower_bounds, upper_bounds

    def _build_obs_space(self):
        """
        Build the observation space.

        Notes
        -----
        1. The returned observation space is normalized so that the environment
        emits observations in the range [-1, 1]. The conversion is handled
        by the environment in `step()`.
        2. For observations that are not

        Returns
        -------
        obs_space : gym.spaces.Box
            The normalized observation space of the environment.
        lower_bounds : numpy.ndarray
            The actual lower bounds on observations emitted by the environment.
        upper_bounds : numpy.ndarray
            The actual upper bounds on observations emitted by the environment.
        obs_space_bounded : bool
            True if the non-scaled observation space is lower and upper bounded.
        """

        network_specs = {k: np.array(v) for k, v in self.network_specs.items()}
        lower_bounds, upper_bounds = [], []

        for name in self.obs_values:

            if name == 'P_BUS':
                lower_bounds.append(network_specs['PMIN_BUS'])
                upper_bounds.append(network_specs['PMAX_BUS'])

            elif name == 'Q_BUS':
                lower_bounds.append(network_specs['QMIN_BUS'])
                upper_bounds.append(network_specs['QMAX_BUS'])

            elif name == 'V_MAGN_BUS':
                lower_bounds.append(network_specs['VMIN_BUS'])
                upper_bounds.append(network_specs['VMAX_BUS'])

            elif name == 'P_DEV':
                lower_bounds.append(network_specs['PMIN_DEV'])
                upper_bounds.append(network_specs['PMAX_DEV'])

            elif name == 'Q_DEV':
                lower_bounds.append(network_specs['QMIN_DEV'])
                upper_bounds.append(network_specs['QMAX_DEV'])

            elif name == 'SOC':
                lower_bounds.append(network_specs['SOC_MIN'])
                upper_bounds.append(network_specs['SOC_MAX'])

            elif name in ['P_BR_F', 'P_BR_T', 'Q_BR_F', 'Q_BR_T', 'I_MAGN_F',
                          'I_MAGN_T', 'S_FLOW']:
                shape = network_specs['RATE'].shape
                lower_bounds.append(- np.inf * np.ones(shape=shape))
                upper_bounds.append(np.inf * np.ones(shape=shape))

            else:
                raise ValueError('The type of observation ' + name
                                 + 'is not supported.')

        lower_bounds = np.concatenate(lower_bounds)
        upper_bounds = np.concatenate(upper_bounds)

        # Check if the non-normalized space is lower and upper bounded.
        actual_space = spaces.Box(low=lower_bounds, high=upper_bounds)
        obs_space_bounded = actual_space.is_bounded()

        if obs_space_bounded:
            obs_space = spaces.Box(low=-np.ones_like(lower_bounds),
                                   high=np.ones_like(upper_bounds))
        else:
            print('Warning: the observation space was not normalized to [-1, 1]'
                  'because it is not bounded.')
            obs_space = actual_space

        return obs_space, lower_bounds, upper_bounds, obs_space_bounded

    def init_dg_load(self, pmax, init_date, delta_t, np_random):

        raise NotImplementedError('The function init_dg_load() should be '
                                  'implemented by the subclass.')

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
            The reward associated with the transition.
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

        # Re-scale the normalized action from [-1, 1] to the actual action
        # space.
        action = utils.linear_scale(action, self.action_space.low,
                                    self.action_space.high, self.action_low,
                                    self.action_high)

        # Get the output of the stochastic processes (vre generation, loads).
        P_loads = [next(load) for load in self.loads]
        self.P_gen_potential = [next(gen) for gen in self.generators]

        # Separate 3 types of actions.
        i = self.action_lengths[0]
        j = i + self.action_lengths[1]
        k = j + self.action_lengths[2]
        P_curt_limit = action[0: i]
        des_alpha = action[i: j]
        Q_storage = action[j: k]

        # Simulate a transition and compute the reward.
        self.state, reward, self.e_loss, self.penalty = \
            self.simulator.transition(P_loads, self.P_gen_potential,
                                      P_curt_limit, des_alpha, Q_storage)
        self.total_reward += reward

        # Create a tuple of observations.
        self.obs = self._get_observations()

        # Linearly scale the observations to be in [-1, 1].
        scaled_obs = utils.linear_scale(self.obs, self.obs_low,
                                        self.obs_high,
                                        self.observation_space.low,
                                        self.observation_space.high)

        # Check if the observation is in the available observation space.
        assert self.observation_space.contains(scaled_obs), "%r (%s) invalid" \
                                                % (scaled_obs, type(scaled_obs))

        # End of episode if maximum number of time steps has been reached.
        self._increment_t()
        if self.end_time is not None and self.time >= self.end_time:
            self.done = True

        # Information returned for debugging.
        info = {'episode': None, 'year': self.year_counter}

        return scaled_obs, reward, self.done, info

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
            obs = np.concatenate(obs).astype(np.float32)
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
        iterators = self.init_dg_load(dev_specs, self.time, self.delta_t,
                                      self.np_random)
        utils.check_load_dg_iterators(iterators, dev_specs)

        # Separate loads and generators.
        self.loads, self.generators = \
            self._separate_load_dg(iterators, [t for (t, p) in dev_specs])

        # Reset the initial SoC of each storage unit.
        soc_start = self.init_soc(self.network_specs['SOC_MAX'])
        utils.check_init_soc(soc_start, self.network_specs['SOC_MAX'])
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

    def _get_dev_specs(self):
        """
        Return the device specs needed to initialize the stochastic processes.

        Returns
        -------
        pmax : list of (int, float)
            A pair (dev_type, pmax) for each device representing a stochastic
            process (e.g. renewable energy resource).
        """

        pmax = []
        for idx, dev_type in enumerate(self.network_specs['DEV_TYPE']):
            p_max = self.network_specs['PMAX_DEV'][idx]
            p_min = self.network_specs['PMIN_DEV'][idx]
            if dev_type == -1:
                pmax.append((dev_type, p_min))
            elif dev_type in [1, 2, 3]:
                pmax.append((dev_type, p_max))

        return pmax

    def _separate_load_dg(self, iterators, dev_types):
        """
        Separate the loads and generators objects modelling stochastic processes.

        Parameters
        ----------
        iterators : list of Iterable
            The passive loads and renewable energy generators.
        dev_types : list of (int, float)
            The specs of each device modelling a stochastic process (see
            `_get_dev_specs()`).

        Returns
        -------
        loads : list of Iterable
            The loads.
        gens : list of Iterable
            The renewable energy generators.
        """

        loads, gens = [], []

        for idx, dev_type in enumerate(dev_types):
            if dev_type == -1:
                loads.append(iterators[idx])
            elif dev_type in [1, 2, 3]:
                gens.append(iterators[idx])
            else:
                raise ValueError('This type of device is not supported.')

        return loads, gens

    def _init_action(self):
        """
        Get the action that doesn't modify the state of the system.

        Returns
        -------
        action : numpy.ndarray
            The action to take which will not modify the state.
        """

        P_curt_bounds = self.action_space.high[0: self.action_lengths[0]]
        action = np.concatenate([P_curt_bounds,
                                 [0] * self.action_lengths[1],
                                 [0] * self.action_lengths[2]])

        return action

    def init_soc(self, soc_max):
        """
        Get the initial state of charge for each storage unit.

        Parameters
        ----------
        soc_max : list of float
            The maximum state of charge of each DES unit.

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