import os

# Root folder path (used in rendering).
ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))

from . import simulator, envs
from gym.envs.registration import register

register(
    id='ANM6Easy-v0',
    entry_point='gym_anm.envs:ANM6Easy',
)
