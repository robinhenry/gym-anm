import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import datetime as dt
from copy import copy
from warnings import warn

from gym_anm.simulator import Simulator
from gym_anm.errors import ObsSpaceError, ObsNotSupportedError
from gym_anm.utils import check_env_args
from gym_anm.constants import STATE_VARIABLES
from gym_anm.simulator.components import StorageUnit, Generator


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
    state_bounds : dict of {str : list}
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

    def __init__(self, network, observation, K, delta_t, gamma, lamb, seed=None):
        """
        Parameters
        ----------
        network : dict of {str : numpy.ndarray}
            The network input dictionary describing the power grid.
        observation : callable or list or str
            The observation space. It can be specified as "state" to construct a
            fully observable environment (o_t = s_t); as a callable function such
            that o_t = observation(s_t); or as a list of tuples (x, y, z) that
            refers to the electrical quantity x (str) at the nodes/branches/devices
            y (list) in unit z (str, optional).
        K : int
            The number of auxiliary variables.
        delta_t : float
            The interval of time between two consecutive time steps (fraction of
            hour).
        gamma : float
            The discount factor in [0, 1].
        lamb : int or float
            The factor multiplying the penalty associated with violating
            operational constraints (used in the reward signal).
        seed : int, optional
            A random seed.
        """

        self.K = K
        self.gamma = gamma
        self.lamb = lamb
        self.delta_t = delta_t

        self.seed(seed)

        # Time variables.
        self.timestep_length = dt.timedelta(minutes=int(60 * delta_t))
        self.year = 2020

        # Initialize the AC power grid simulator.
        self.simulator = Simulator(network, self.delta_t, self.lamb)

        # Check the arguments provided.
        check_env_args(K, delta_t, lamb, gamma, observation,
                       self.simulator.state_bounds)

        # Variables to include in state vectors.
        self.state_values = [('dev_p', 'all', 'MW'), ('dev_q', 'all', 'MVAr'),
                             ('des_soc', 'all', 'MWh'), ('gen_p_max', 'all', 'MW'),
                             ('aux', 'all', None)]

        # Build action space.
        self.action_space = self._build_action_space()

        # Build observation space.
        self.obs_values = self._build_observation_space(observation)
        self.observation_space = self.observation_bounds()

    def init_state(self):
        raise NotImplementedError

    def next_vars(self, s_t):
        raise NotImplementedError

    def observation_bounds(self):
        """
        Builds the observation space of the environment.

        If the observation space is specified as a callable object, then its
        bounds are set to (- np.inf, np.inf)^{N_o} by default (this is done
        during the `reset()` call, as the size of observation vectors is not
        known before then. Alternatively, the user can specify its own bounds
        by overwriting this function in the new environment.

        Returns
        -------
        gym.spaces.Box or None
            The bounds of the observation space.
        """
        lower_bounds, upper_bounds = [], []

        if self.obs_values is None:
            warn('The observation space is unbounded.')
            # In this case, the size of the obs space is obtained after the
            # environment has been reset. See `reset()`.
            return None

        else:
            bounds = self.simulator.state_bounds
            for key, nodes, unit in self.obs_values:
                for n in nodes:
                    lower_bounds.append(bounds[key][n][unit][0])
                    upper_bounds.append(bounds[key][n][unit][1])

        space = spaces.Box(low=np.array(lower_bounds),
                           high=np.array(upper_bounds))

        return space

    def reset(self):
        """
        Reset the environment.

        Returns
        -------
        obs : numpy.ndarray
            The initial observation vector.
        """

        self.done = False
        self.render_mode = None
        self.timestep = 0
        self.total_disc_return = 0.

        # Initialize the state.
        self.state = self.init_state()

        # Apply the initial state to the simulator.
        self.simulator.reset(self.state)
        self.state = self._construct_state()  # in case the original state was infeasible.

        # Construct the initial observation vector.
        obs = self.observation(self.state)

        # Update the observation space bounds if required.
        if self.observation_space is None:
            self.observation_space = spaces.Box(low=-np.ones(len(obs)) * np.inf,
                                                high=np.ones(len(obs)) * np.inf)

        err_msg = "Observation %r (%s) invalid." % (obs, type(obs))
        assert self.observation_space.contains(obs), err_msg

        return obs

    def observation(self, s_t):
        """
        Returns the observation vector corresponding to the current state `s_t`.

        Alternatively, this function can be overwritten in custom environments.

        Parameters
        ----------
        s_t : numpy.ndarray
            The current state vector `s_t`.

        Returns
        -------
        numpy.ndarray
            The corresponding observation vector `o_t`.
        """
        return self._extract_state_variables(self.obs_values)

    def step(self, action):
        """
        Take a control action and transition from a state s_t to a state s_{t+1}.

        Parameters
        ----------
        action : numpy.ndarray
            The action vector `a_t` taken by the agent.

        Returns
        -------
        obs : numpy.ndarray
            The observation vector `o_{t+1}`.
        reward : float
            The reward associated with the transition `r_t`.
        done : bool
            True if the episode is over, False otherwise. Always False in
            `gym-anm` environments.
        info : dict
            A dictionary with further information (used for debugging).
        """

        err_msg = "Action %r (%s) invalid." % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # 1a. Sample the internal stochastic variables.
        vars = self.next_vars(self.state)
        P_load = vars[:self.simulator.N_load]
        P_pot = vars[self.simulator.N_load: self.simulator.N_load + self.simulator.N_non_slack_gen]
        aux = vars[self.simulator.N_load + self.simulator.N_non_slack_gen:]
        assert len(aux) == self.K

        # 2. Extract the different actions from the action vector.
        P_set_points = {}
        Q_set_points = {}
        gen_slack_ids = [i for i, dev in self.simulator.devices
                             if isinstance(dev, Generator) and not dev.is_slack]
        des_ids = [i for i, dev in self.simulator.devices if isinstance(dev, StorageUnit)]
        N_gen = len(gen_slack_ids)
        N_des = len(des_ids)

        for a, dev_id in zip(action[:N_gen], gen_slack_ids):
            P_set_points[dev_id] = a
        for a, dev_id in zip(action[N_gen: 2 * N_gen], gen_slack_ids):
            Q_set_points[dev_id] = a
        for a, dev_id in zip(action[2 * N_gen: 2 * N_gen + N_des], des_ids):
            P_set_points[dev_id] = a
        for a, dev_id in zip(action[2 * N_gen + N_des:], des_ids):
            Q_set_points[dev_id] = a

        # 3. Apply the action in the simulator.
        _, r, self.e_loss, self.penalty = \
            self.simulator.transition(P_load, P_pot, P_set_points, Q_set_points)

        # 4. Construct the state and observation vector.
        self.state = self._construct_state()
        obs = self.observation(self.state)

        err_msg = "Observation %r (%s) invalid." % (obs, type(obs))
        assert self.observation_space.contains(obs), err_msg

        # 5. Update the discounted return.
        self.total_disc_return += self.gamma ** self.timestep * r
        self.timestep += 1

        return obs, r, self.done, {}


    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self, seed=None):
        """Seed the random number generator. """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _build_action_space(self):
        """
        Build the available loose action space `\mathcal A`.

        Returns
        -------
        space : gym.spaces.Box
            The action space of the environment.
        """

        P_gen_bounds, Q_gen_bounds, P_des_bounds, Q_des_bounds = \
            self.simulator.get_action_space()

        lower_bounds, upper_bounds = [], []

        for dev_id in sorted(P_gen_bounds.keys()):
            upper_bounds.append(P_gen_bounds[dev_id][0])
            lower_bounds.append(P_gen_bounds[dev_id][1])

        for dev_id in sorted(Q_gen_bounds.keys()):
            upper_bounds.append(Q_gen_bounds[dev_id][0])
            lower_bounds.append(Q_gen_bounds[dev_id][1])

        for dev_id in sorted(P_des_bounds.keys()):
            upper_bounds.append(P_des_bounds[dev_id][0])
            lower_bounds.append(Q_des_bounds[dev_id][1])

        space = spaces.Box(low=np.array(lower_bounds),
                           high=np.array(upper_bounds))

        return space

    def _build_observation_space(self, observation):
        """Handles the different ways of specifying an observation space."""

        # Case 1: environment is fully observable.
        if isinstance(observation, str) and observation == 'state':
            obs_values = self.state_values

        # Case 2: observation space is provided as a list.
        elif isinstance(observation, list):
            obs_values = copy(observation)
            # Add default units when none is provided.
            for idx, o in enumerate(obs_values):
                if len(o) == 2:
                    obs_values[idx] = tuple(list(o) + STATE_VARIABLES[o[0]][0])

        # Case 3: observation space is provided as a callable object.
        elif callable(observation):
            obs_values = None
            self.observation = observation

        else:
            raise ObsSpaceError()

        # Transform the 'all' option into a list of bus/branch/device IDs.
        if obs_values is not None:
            for idx, o in enumerate(obs_values):
                if isinstance(o[1], str) and o[1] == 'all':
                    if 'bus' in o[0]:
                        ids = list(self.simulator.buses.keys())
                    elif 'dev' in o[0]:
                        ids = list(self.simulator.devices.keys())
                    elif 'des' in o[0]:
                        ids = [i for i, d in self.simulator.devices.items() if isinstance(d, StorageUnit)]
                    elif 'gen' in o[0]:
                        ids = [i for i, d in self.simulator.devices.items() if isinstance(d, Generator) and not d.is_slack]
                    elif 'branch' in o[0]:
                        ids = list(self.simulator.branches.keys())
                    elif o[0] == 'aux':
                        ids = list(range(0, self.K))
                    else:
                        raise ObsNotSupportedError(o[0], STATE_VARIABLES.keys())

                    obs_values[idx] = (o[0], ids, o[2])

        return obs_values



    def _construct_state(self):
        """
        Construct the state vector `s_t`.

        Returns
        -------
        s_t : numpy.ndarray
            The state vector
        """
        return self._extract_state_variables(self.state_values)

    def _extract_state_variables(self, values):
        """
        Extract variables used in state and observation vectors from the simulator.

        Parameters
        ----------
        values : list of tuple of (str, list, str)
            The variables to extract as tuples, where each tuple (i, j, k) refers
            to variable i at the nodes/branches/devices listed in j, using unit
            k.

        Returns
        -------
        numpy.ndarray
            The vector of extracted state variables.
        """

        full_state = self.simulator.state

        vars = []
        for value in values:
            if value[0] in full_state.keys():
                o = full_state[value[0]][value[2]][value[1]]
            elif value[0] == 'aux':
                o = self.state[value[1] - self.K]
            else:
                raise ObsNotSupportedError(value[0], STATE_VARIABLES.keys())

            vars.append(o)

        return vars
