from .smartgrid_env import SmartGridEnv
from .cases import case2, case6


class SmartGridEnv2(SmartGridEnv):
    def __init__(self):
        case = case2.load()
        super().__init__(case)

class SmartGridEnv6(SmartGridEnv):
    def __init__(self):
        case = case6.load()
        super().__init__(case)