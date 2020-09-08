from . import simulator, envs
from gym.envs.registration import register
from .dc_opf.dc_opf import DCOPFAgent
from .dc_opf.mpc_anm6_easy import MPCAgentANM6Easy

register(
    id='ANM6Easy-v0',
    entry_point='gym_anm.envs:ANM6Easy',
)

register(id='ANM6Partial-v0', entry_point='gym_anm.envs:ANM6Partial')
