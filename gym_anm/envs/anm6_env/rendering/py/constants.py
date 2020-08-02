import os

from gym_anm.utils import get_package_root


RENDERING_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDERING_LOGS = os.path.join(RENDERING_FOLDER, 'logs')
RENDERING_RELATIVE_PATH = os.path.join('envs', 'anm6_env', 'rendering')
ROOT_FOLDER = get_package_root()
