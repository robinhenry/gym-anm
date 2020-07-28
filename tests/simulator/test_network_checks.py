import unittest
import os
import numpy as np

from gym_anm.simulator.components.errors import *

class TestNetworkChecks(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # Set the working directory to the root.

    def test_baseMVA(self):
        pass

    def test_bus_id_duplicates(self):
        pass

    def test_dev_id_duplicates(self):
        pass

    def test_branch_duplicates(self):
        pass

    def test_no_slack_bus(self):
        pass

    def test_no_slack_dev(self):
        pass

    def test_too_many_slack_buses(self):
        pass

    def test_too_many_slack_devs(self):
        pass

    def test_slack_dev_not_at_slack_bus(self):
        pass


if __name__ == '__main__':
    unittest.main()
