import os
from gym.envs.registration import register

register(
    id='acgrid-v0',
    entry_point='gym_smartgrid.envs:FooEnv',
)

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
RENDERING_FOLDER = os.path.join(ROOT_FOLDER, 'rendering')
RENDERING_LOGS = os.path.join(RENDERING_FOLDER, 'logs')
ENV_FILES = {'case': 'case.py',
             'network': 'network.svg',
             'svgLabels': 'svgLabels.js'}
WEB_FILES = {'index': 'index.html'}
