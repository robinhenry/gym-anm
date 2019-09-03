import os
from gym.envs.registration import register

register(
    id='acgrid-v0',
    entry_point='gym_anm.envs:FooEnv',
)

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
