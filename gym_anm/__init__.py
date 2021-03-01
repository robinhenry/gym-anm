from . import simulator, envs
from gym.envs.registration import register
from .agents import DCOPFAgent
from .agents import MPCAgentANM6Easy
from .envs import ANMEnv

register(
    id='ANM6Easy-v0',
    entry_point='gym_anm.envs:ANM6Easy',
)
