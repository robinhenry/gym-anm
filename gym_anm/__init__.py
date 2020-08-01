from gym.envs.registration import register
from . import simulator, envs, utils, errors


register(
    id='ANM6Easy-v0',
    entry_point='gym_anm.envs:ANM6Easy',
)
