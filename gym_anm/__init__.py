from . import simulator, envs
from gym.envs.registration import register
from .dc_opf import DCOPFAgent
from .dc_opf import MPCAgentANM6Easy

register(
    id='ANM6Easy-v0',
    entry_point='gym_anm.envs:ANM6Easy',
)
