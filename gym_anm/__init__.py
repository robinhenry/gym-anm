"""A package for designing RL ANM tasks in power grids."""

from gym.envs.registration import register

from . import simulator, envs
from .agents import MPCAgent, MPCAgentANM6Easy
from .envs import ANMEnv

register(
    id='ANM6Easy-v0',
    entry_point='gym_anm.envs:ANM6Easy',
)
