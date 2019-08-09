import os

from gym_smartgrid.envs import SmartGridEnv
from .case6 import load

class SmartGridEnv6(SmartGridEnv):
    def __init__(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        svg_data = {'network': os.path.join(dir_path, 'case6.svg')}
        svg_data['labels'] = os.path.join(dir_path, 'svgLabels.js')

        super().__init__(load, svg_data)
