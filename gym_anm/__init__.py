"""A package for designing RL ANM tasks in power grids."""

from gym.envs.registration import register

from .agents import MPCAgentPerfect, MPCAgentConstant
from .envs import ANMEnv

register(
    id='ANM6Easy-v0',
    entry_point='gym_anm.envs:ANM6Easy',
)
