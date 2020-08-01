import os
from gym.envs.registration import register

register(
    id='ANM6Easy-v0',
    entry_point='gym_anm.envs:ANM6Easy',
)

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
