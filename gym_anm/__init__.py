from gym.envs.registration import register
import os


ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))

register(
    id='ANM6Easy-v0',
    entry_point='gym_anm.envs:ANM6Easy',
)
