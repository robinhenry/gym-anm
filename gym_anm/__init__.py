import os
from gym.envs.registration import register

register(
    id='ANM6-Easy-v0',
    entry_point='gym_anm.envs:ANM6Easy',
)
register(
    id='ANM6-Hard-v0',
    entry_point='gym_anm.envs:ANM6Hard',
)

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
