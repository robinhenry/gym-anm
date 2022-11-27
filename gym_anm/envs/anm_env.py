"""The base class for :code:`gym-anm` environments."""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from logging import getLogger
from copy import deepcopy
import warnings
from scipy.sparse.linalg import MatrixRankWarning

from ..simulator import Simulator
from ..errors import ObsSpaceError, ObsNotSupportedError, EnvInitializationError, EnvNextVarsError
from .utils import check_env_args
from ..simulator.components.constants import STATE_VARIABLES
from ..simulator.components import StorageUnit, Generator, Load


logger = getLogger(__file__)


class ANMEnv(gym.Env):
    """
    The base class for :code:`gym-anm` environments.

    Attributes
    ----------
    K : int
        The number of auxiliary variables.
    gamma : float
        The fixed discount factor in [0, 1].
    lamb : int or float
        The factor multiplying the penalty associated with violating
        operational constraints (used in the reward signal).
    delta_t : float
        The interval of time between two consecutive time steps (fraction of
        hour).
    simulator : :py:class:`gym_anm.simulator.simulator.Simulator`
        The electricity distribution network simulator.
    state_values : list of tuple of str
        The electrical quantities to include in the state vectors. Each tuple
        (x, y, z) refers to quantity x at nodes/devices/branches y, using units
        z.
    state_N : int
        The number of state variables.
    action_space : gym.spaces.Box
        The action space from which the agent can select actions.
    obs_values : list of str or None
        Similarly to :py:obj:`state_values`, the values to include in the observation
        vectors. If a customized :py:func:`observation()` function is provided, :py:obj:`obs_values`
        is None.
    observation_space : gym.spaces.Box
        The observation space from which observation vectors are constructed.
    observation_N : int
        The number of observation variables.
    done : bool
        True if a terminal state has been reached (if the network collapsed);
        False otherwise.
    render_mode : str
        The rendering mode. See :py:func:`render()`.
    timestep : int
        The current timestep.
    state : numpy.ndarray
        The current state vector :math:`s_t`.
    e_loss : float
        The energy loss during the last transition (part of the reward signal).
    penalty : float
        The penalty associated with violating operational constraints during the
        last transition (part of the reward signal).
    costs_clipping : tuple of float
        The clipping values for the costs (- rewards), where :py:obj:`costs_clipping[0]` is
        the clipping value for the absolute energy loss and :py:obj:`costs_clipping[1]` is
        the clipping value for the constraint violation penalty.
    pfe_converged : bool
        True if the last transition converged to a load flow solution (i.e.,
        the network is stable); False otherwise.
    np_random : numpy.random.RandomState
        The random state/seed of the environment.
    """

    def __init__(self, network, observation, K, delta_t, gamma, lamb, aux_bounds=None, costs_clipping=None, seed=None):
        """
        Parameters
        ----------
        network : dict of {str : numpy.ndarray}
            The network input dictionary describing the power grid.
        observation : callable or list or str
            The observation space. It can be specified as "state" to construct a
            fully observable environment (:math:`o_t = s_t`); as a callable function such
            that :math:`o_t = observation(s_t)`; or as a list of tuples (x, y, z) that
            refer to the electrical quantities x (str) at the nodes/branches/devices
            y (list or 'all') in unit z (str, optional).
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
        aux_bounds : numpy.ndarray, optional
            The bounds on the auxiliary internal variables as a 2D array where
            the :math:`k^{th}`-1 auxiliary variable is bounded by
            :py:obj:`[aux_bounds[k, 0], aux_bounds[k, 1]]`. This can be useful if auxiliary
            variables are to be included in the observation vectors and a bounded
            observation space is desired.
        costs_clipping : tuple of float, optional
            The clipping values for the costs in the reward signal, where element
            0 is the clipping value for the energy loss cost and element 1 is the
            clipping value for the constraint-violation penalty (e.g., (1, 100)).
        seed : int, optional
            A random seed.
        """

        self.seed(seed)

        self.K = K
        self.gamma = gamma
        self.lamb = lamb
        self.delta_t = delta_t
        self.aux_bounds = aux_bounds

        # Do not clip costs if unspecified.
        if costs_clipping is None:
            c1, c2 = np.inf, np.inf
        else:
            c1 = np.inf if costs_clipping[0] is None else costs_clipping[0]
            c2 = np.inf if costs_clipping[1] is None else costs_clipping[1]
        self.costs_clipping = (c1, c2)

        # Initialize the AC power grid simulator.
        self.simulator = Simulator(network, self.delta_t, self.lamb)

        # Check the arguments provided.
        check_env_args(K, delta_t, lamb, gamma, observation, aux_bounds, self.simulator.state_bounds)

        # Variables to include in state vectors.
        self.state_values = [
            ("dev_p", "all", "MW"),
            ("dev_q", "all", "MVAr"),
            ("des_soc", "all", "MWh"),
            ("gen_p_max", "all", "MW"),
            ("aux", "all", None),
        ]
        self.state_values = self._expand_all_ids(self.state_values)
        self.state_N = sum(len(s[1]) for s in self.state_values)

        # Build action space.
        self.action_space = self._build_action_space()

        # Build observation space.
        self.obs_values = self._build_observation_space(observation)
        self.observation_space = self.observation_bounds()
        if self.observation_space is not None:
            self.observation_N = self.observation_space.shape[0]

    def init_state(self):
        """
        Sample an initial state :math:`s_0`.

        For reproducibility, the RandomState :py:obj:`self.np_random` should be used to
        generate random numbers.

        Returns
        -------
        numpy.ndarray
            An initial state vector :math:`s_0`.
        """
        raise NotImplementedError

    def next_vars(self, s_t):
        """
        Sample internal variables.

        Parameters
        ----------
        s_t : numpy.ndarray
            The current state vector :math:`s_t`.

        Returns
        -------
        numpy.ndarray
            The internal variables for the next timestep, following the structure
            :math:`[P_l, P_g^{(max)}, aux^{(k)}]`, where :math:`P_l` contains the load
            injections (ordered by device ID), :math:`P_g^{(max)}` the maximum generation
            from non-slack generators (ordered by device ID), and :math:`aux^{(k)} `the
            auxiliary variables. The vector shape should be
            :code:`(N_load + (N_generators-1) + K,)`.
        """
        raise NotImplementedError

    def observation_bounds(self):
        """
        Builds the observation space of the environment.

        If the observation space is specified as a callable object, then its
        bounds are set to :code:`(- np.inf, np.inf)^{N_o}` by default (this is done
        during the :py:func:`reset()` call, as the size of observation vectors is not
        known before then). Alternatively, the user can specify their own bounds
        by overwriting this function in a subclass.

        Returns
        -------
        gym.spaces.Box or None
            The bounds of the observation space.
        """
        lower_bounds, upper_bounds = [], []

        if self.obs_values is None:
            logger.warning("The observation space is unbounded.")
            # In this case, the size of the obs space is obtained after the
            # environment has been reset. See `reset()`.
            return None

        else:
            bounds = self.simulator.state_bounds
            for key, nodes, unit in self.obs_values:
                for n in nodes:
                    if key == "aux":
                        if self.aux_bounds is not None:
                            lower_bounds.append(self.aux_bounds[n][0])
                            upper_bounds.append(self.aux_bounds[n][1])
                        else:
                            lower_bounds.append(-np.inf)
                            upper_bounds.append(np.inf)
                    else:
                        lower_bounds.append(bounds[key][n][unit][0])
                        upper_bounds.append(bounds[key][n][unit][1])

        space = spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds), dtype=np.float64)

        return space

    def reset(self):
        """
        Reset the environment.

        If the observation space is provided as a callable object but the
        :py:func:`observation_bounds()` method is not overwritten, then the bounds on the
        observation space are set to :code:`(- np.inf, np.inf)` here (after the size of the
        observation vectors is known).

        Returns
        -------
        obs : numpy.ndarray
            The initial observation vector.
        """

        self.done = False
        self.render_mode = None
        self.timestep = 0
        self.e_loss = 0.0
        self.penalty = 0.0

        # Initialize the state.
        init_state_found = False
        n_init_states = 0
        n_init_states_max = 100
        while not init_state_found:
            n_init_states += 1
            self.state = self.init_state()

            # Check s_0 has the correct size.
            expected = 2 * self.simulator.N_device + self.simulator.N_des + self.simulator.N_non_slack_gen + self.K
            if self.state.size != expected:
                msg = "Expected size of initial state s0 is %d but actual is %d" % (expected, self.state.size)
                raise EnvInitializationError(msg)

            # Apply the initial state to the simulator.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", MatrixRankWarning)
                init_state_found = self.simulator.reset(self.state)

            if n_init_states == n_init_states_max:
                msg = "No non-terminal state found out of %d initial states for " "environment %s" % (
                    n_init_states_max,
                    self.__name__,
                )
                raise EnvInitializationError(msg)

        # Reconstruct the sate vector in case the original state was infeasible.
        self.state = self._construct_state()

        # Construct the initial observation vector.
        obs = self.observation(self.state)

        # Update the observation space bounds if required.
        if self.observation_space is None:
            self.observation_space = spaces.Box(low=-np.ones(len(obs)) * np.inf, high=np.ones(len(obs)) * np.inf)
            self.observation_N = self.observation_space.shape[0]

        err_msg = "Observation %r (%s) invalid." % (obs, type(obs))
        assert self.observation_space.contains(obs), err_msg

        # Cast state and obs vectors to 0 (arbitrary) if a terminal state has
        # been reached.
        if self.done:
            self.state = self._terminal_state(self.state_N)
            obs = self._terminal_state(self.observation_N)

        return obs

    def observation(self, s_t):
        """
        Returns the observation vector corresponding to the current state :math:`s_t`.

        Alternatively, this function can be overwritten in customized environments.

        Parameters
        ----------
        s_t : numpy.ndarray
            The current state vector :math:`s_t`.

        Returns
        -------
        numpy.ndarray
            The corresponding observation vector :math:`o_t`.
        """
        obs = self._extract_state_variables(self.obs_values)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def step(self, action):
        """
        Take a control action and transition from state :math:`s_t` to state :math:`s_{t+1}`.

        Parameters
        ----------
        action : numpy.ndarray
            The action vector :math:`a_t` taken by the agent.

        Returns
        -------
        obs : numpy.ndarray
            The observation vector :math:`o_{t+1}`.
        reward : float
            The reward associated with the transition :math:`r_t`.
        done : bool
            True if a terminal state has been reached; False otherwise.
        info : dict
            A dictionary with further information (used for debugging).
        """

        err_msg = "Action %r (%s) invalid." % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # 0. Remain in a terminal state and output reward=0 if the environment
        # has already reached a terminal state.
        if self.done:
            obs = self._terminal_state(self.observation_N)
            return obs, 0.0, self.done, {}

        # 1a. Sample the internal stochastic variables.
        vars = self.next_vars(self.state)
        expected_size = self.simulator.N_load + self.simulator.N_non_slack_gen + self.K
        if vars.size != expected_size:
            msg = "Next vars vector has size %d but expected is %d" % (vars.size, expected_size)
            raise EnvNextVarsError(msg)

        P_load = vars[: self.simulator.N_load]
        P_pot = vars[self.simulator.N_load : self.simulator.N_load + self.simulator.N_non_slack_gen]
        aux = vars[self.simulator.N_load + self.simulator.N_non_slack_gen :]
        err_msg = "Only {} auxiliary variables are generated, but K={} are " "expected.".format(len(aux), self.K)
        assert len(aux) == self.K, err_msg

        # 1b. Convert internal variables to dictionaries.
        load_idx, gen_idx = 0, 0
        P_load_dict, P_pot_dict = {}, {}
        for dev_id, dev in self.simulator.devices.items():
            if isinstance(dev, Load):
                P_load_dict[dev_id] = P_load[load_idx]
                load_idx += 1
            elif isinstance(dev, Generator) and not dev.is_slack:
                P_pot_dict[dev_id] = P_pot[gen_idx]
                gen_idx += 1

        # 2. Extract the different actions from the action vector.
        P_set_points = {}
        Q_set_points = {}
        gen_non_slack_ids = [
            i for i, dev in self.simulator.devices.items() if isinstance(dev, Generator) and not dev.is_slack
        ]
        des_ids = [i for i, dev in self.simulator.devices.items() if isinstance(dev, StorageUnit)]
        N_gen = len(gen_non_slack_ids)
        N_des = len(des_ids)

        for a, dev_id in zip(action[:N_gen], gen_non_slack_ids):
            P_set_points[dev_id] = a
        for a, dev_id in zip(action[N_gen : 2 * N_gen], gen_non_slack_ids):
            Q_set_points[dev_id] = a
        for a, dev_id in zip(action[2 * N_gen : 2 * N_gen + N_des], des_ids):
            P_set_points[dev_id] = a
        for a, dev_id in zip(action[2 * N_gen + N_des :], des_ids):
            Q_set_points[dev_id] = a

        # 3a. Apply the action in the simulator.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MatrixRankWarning)
            _, r, e_loss, penalty, pfe_converged = self.simulator.transition(
                P_load_dict, P_pot_dict, P_set_points, Q_set_points
            )

            # A terminal state has been reached if no solution to the power
            # flow equations is found.
            self.done = not pfe_converged

        # 3b. Clip the reward.
        if not self.done:
            self.e_loss = np.sign(e_loss) * np.clip(np.abs(e_loss), 0, self.costs_clipping[0])
            self.penalty = np.clip(penalty, 0, self.costs_clipping[1])
            r = -(self.e_loss + self.penalty)
        else:
            # Very large reward if a terminal state has been reached.
            r = -self.costs_clipping[1] / (1 - self.gamma)
            self.e_loss = self.costs_clipping[0]
            self.penalty = self.costs_clipping[1]

        # 4. Construct the state and observation vector.
        if not self.done:
            for k in range(self.K):
                self.state[k - self.K] = aux[k]
            self.state = self._construct_state()
            obs = self.observation(self.state)

            err_msg = "Observation %r (%s) invalid." % (obs, type(obs))
            assert self.observation_space.contains(obs), err_msg

        # Cast state and obs vectors to 0 (arbitrary) if a terminal state is
        # reached.
        else:
            self.state = self._terminal_state(self.state_N)
            obs = self._terminal_state(self.observation_N)

        # 5. Update the timestep.
        self.timestep += 1

        return obs, r, self.done, {}

    def render(self, mode="human"):
        """
        Update the rendering of the environment (to be overwritten).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

    def close(self):
        """
        Close the rendering of the environment (to be overwritten).

        Raises
        -------
        NotImplementedError
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Seed the random number generator."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _build_action_space(self):
        """
        Build the available loose action space :math:`\mathcal A`.

        Returns
        -------
        gym.spaces.Box
            The action space of the environment.
        """

        P_gen_bounds, Q_gen_bounds, P_des_bounds, Q_des_bounds = self.simulator.get_action_space()

        lower_bounds, upper_bounds = [], []
        for x in [P_gen_bounds, Q_gen_bounds, P_des_bounds, Q_des_bounds]:
            for dev_id in sorted(x.keys()):
                lower_bounds.append(x[dev_id][0])
                upper_bounds.append(x[dev_id][1])

        space = spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds), dtype=np.float64)

        return space

    def _build_observation_space(self, observation):
        """Handles the different ways of specifying an observation space."""

        # Case 1: environment is fully observable.
        if isinstance(observation, str) and observation == "state":
            obs_values = deepcopy(self.state_values)

        # Case 2: observation space is provided as a list.
        elif isinstance(observation, list):
            obs_values = deepcopy(observation)
            # Add default units when none is provided.
            for idx, o in enumerate(obs_values):
                if len(o) == 2:
                    obs_values[idx] = tuple(list(o) + [STATE_VARIABLES[o[0]][0]])

        # Case 3: observation space is provided as a callable object.
        elif callable(observation):
            obs_values = None
            self.observation = observation

        else:
            raise ObsSpaceError()

        # Transform the 'all' option into a list of bus/branch/device IDs.
        return self._expand_all_ids(obs_values)

    def _expand_all_ids(self, values):
        """Helper function to translate the 'all' option in a list of IDS."""

        # Transform the 'all' option into a list of bus/branch/device IDs.
        if values is not None:
            for idx, o in enumerate(values):
                if isinstance(o[1], str) and o[1] == "all":
                    if "bus" in o[0]:
                        ids = list(self.simulator.buses.keys())
                    elif "dev" in o[0]:
                        ids = list(self.simulator.devices.keys())
                    elif "des" in o[0]:
                        ids = [i for i, d in self.simulator.devices.items() if isinstance(d, StorageUnit)]
                    elif "gen" in o[0]:
                        ids = [
                            i for i, d in self.simulator.devices.items() if isinstance(d, Generator) and not d.is_slack
                        ]
                    elif "branch" in o[0]:
                        ids = list(self.simulator.branches.keys())
                    elif o[0] == "aux":
                        ids = list(range(0, self.K))
                    else:
                        raise ObsNotSupportedError(o[0], STATE_VARIABLES.keys())

                    values[idx] = (o[0], ids, o[2])

        return values

    def _construct_state(self):
        """
        Construct the state vector :math:`s_t`.

        Returns
        -------
        s_t : numpy.ndarray
            The state vector.
        """
        return self._extract_state_variables(self.state_values)

    def _extract_state_variables(self, values):
        """
        Extract variables used in state and observation vectors from the simulator.

        Parameters
        ----------
        values : list of tuples of (str, list, str)
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
            for idx in value[1]:
                if value[0] in full_state.keys():
                    o = full_state[value[0]][value[2]][idx]
                elif value[0] == "aux":
                    o = self.state[idx - self.K]
                else:
                    raise ObsNotSupportedError(value[0], STATE_VARIABLES.keys())
                vars.append(o)

        return np.array(vars)

    def _terminal_state(self, n):
        """
        Return a 0-vector (arbitrarily chosen as terminal state).

        Parameters
        ----------
        n : int
            The length of the vector to return.

        Returns
        -------
        numpy.ndarray
            A zero-vector of size n.
        """
        return np.zeros(n)
