import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from gym_smartgrid.envs import simulator
from gym_smartgrid.envs.utils import init_load, init_vre
from gym_smartgrid.envs.cases import case2


class SmartGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, case):

        self.t_init = 0

        # Set random seed.
        self.seed()

        # Initialize AC power grid simulator.
        self.simulator = simulator.Simulator(case, self.np_random)

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

        self.generators = init_vre(wind_pmax, solar_pmax,
                                   self.simulator.DELTA_T, self.t_init)

        # Initialize load stochastic processes.
        self.loads = init_load(load_pmax, self.simulator.DELTA_T, self.np_random)

        self.timestep = 0
        self.total_reward = 0.
        self.done = False
        self.max_episode_steps = 4 * 24 * 31 * 12 # 1 year with 15min timesteps.

    def step(self, action):

        # Check if the action is in the available action space.
        assert self.action_space.contains(action), "%r (%s) invalid" \
                                                   % (action, type(action))
        self.timestep += 1

        # Get the output of the stochastic processes (vre generation, loads).
        P_loads = self.loads.next(self.timestep)
        P_gens = self.generators.next(self.timestep)

        # Simulate a transition and compute the reward.
        reward = self.simulator.transition(P_loads, P_gens, *action)
        self.total_reward += reward

        # Create a (4,) tuple of observations.
        obs = self._get_obs()

        # End of episode if maximum number of timesteps has been reached.
        if self.timestep >= self.max_episode_steps:
            self.done = True

        # Information returned for debugging.
        info = None

        return obs, reward, self.done, info

    def _get_obs(self):
        # Create a (4,) tuple of observations.
        obs = np.array(self.simulator.P), np.array(self.simulator.Q), \
              np.array(self.simulator.I_br_magn), np.array(self.simulator.SoC)
        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.timestep = 0
        self.total_reward = 0.
        self.simulator.reset()
        obs = self._get_obs()

        return obs

    def render(self, mode='human', close=False):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class SmartGridEnv2(SmartGridEnv):
    def __init__(self):
        case = case2.load()
        super().__init__(case)

