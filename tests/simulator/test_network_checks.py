import unittest
import os
import numpy as np

from gym_anm.simulator.components.errors import *


@unittest.skip('Not implemented yet.')
class TestNetworkChecks(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # Set the working directory to the root.

    def test_baseMVA(self):
        raise NotImplementedError()

    def test_bus_id_duplicates(self):
        raise NotImplementedError()

    def test_dev_id_duplicates(self):
        raise NotImplementedError()

    def test_branch_duplicates(self):
        raise NotImplementedError()

    def test_no_slack_bus(self):
        raise NotImplementedError()

    def test_no_slack_dev(self):
        raise NotImplementedError()

    def test_too_many_slack_buses(self):
        raise NotImplementedError()

    def test_too_many_slack_devs(self):
        raise NotImplementedError()

    def test_slack_dev_not_at_slack_bus(self):
        raise NotImplementedError()


if __name__ == '__main__':
    unittest.main()
