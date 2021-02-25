import unittest
import os
import numpy as np

from gym_anm.simulator import Simulator
from gym_anm.simulator.components.errors import *


# @unittest.skip('Not implemented yet.')
class TestNetworkChecks(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # Set the working directory to the root.

    def test_baseMVA(self):
        for baseMVA in [-1, 0]:
            network = {
                'baseMVA': baseMVA,
                'bus': np.array([[0, 0, 50, 1.1, 0.9],
                                 [1, 1, 100, 1., 1.]]),
                'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90]]),
                'device': np.array([
                    [1, 0, -1, 0.2, 0, -10, None, None, None, None, None, None, None, None, None],
                    [0, 1, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None]])
            }

            with self.assertRaises(BaseMVAError):
                Simulator(network, 1, 100)

    def test_bus_id_duplicates(self):
        network = {
            'baseMVA': 100,
            'bus': np.array([[0, 0, 50, 1.1, 0.9],
                             [1, 1, 100, 1., 1.],
                             [1, 1, 100, 1., 1.]]),
            'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90]]),
            'device': np.array([
                [0, 0, 0, 0.2, 0, -10, None, None, None, None, None, None, None, None, None],
                [1, 1, -1, None, 200, -200, 200, -200, None, None, None, None, None, None, None]])
        }

        with self.assertRaises(BusSpecError):
            Simulator(network, 1, 100)

    def test_dev_id_duplicates(self):
        network = {
            'baseMVA': 100,
            'bus': np.array([[0, 0, 50, 1.1, 0.9],
                             [1, 1, 100, 1., 1.]]),
            'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90]]),
            'device': np.array([
                [0, 0, 0, 0.2, 0, -10, None, None, None, None, None, None, None, None, None],
                [0, 1, -1, None, 200, -200, 200, -200, None, None, None, None, None, None, None]])
        }
        with self.assertRaises(DeviceSpecError):
            Simulator(network, 1, 100)

    def test_branch_duplicates(self):
        network = {
            'baseMVA': 100,
            'bus': np.array([[0, 0, 50, 1.1, 0.9],
                             [1, 1, 100, 1., 1.]]),
            'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90],
                                [0, 1, 0.4, 0.5, 0.6, 20, 2, 0]]),
            'device': np.array([
                [0, 0, 0, 0.2, 0, -10, None, None, None, None, None, None, None, None, None],
                [1, 1, -1, None, 200, -200, 200, -200, None, None, None, None, None, None, None]])
        }
        with self.assertRaises(BranchSpecError):
            Simulator(network, 1, 100)

    def test_non_existing_bus_in_branch(self):
        network = {
            'baseMVA': 100,
            'bus': np.array([[0, 0, 50, 1.1, 0.9],
                             [1, 1, 100, 1., 1.]]),
            'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90],
                                [1, 2, 0.4, 0.5, 0.6, 20, 2, 0]]),
            'device': np.array([
                [0, 0, 0, 0.2, 0, -10, None, None, None, None, None, None, None, None, None],
                [1, 1, -1, None, 200, -200, 200, -200, None, None, None, None, None, None, None]])
        }
        with self.assertRaises(BranchSpecError):
            Simulator(network, 1, 100)

    def test_no_slack_bus(self):
        network = {
            'baseMVA': 100,
            'bus': np.array([[0, 1, 50, 1.1, 0.9]]),
            'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90]]),
            'device': np.array([
                [0, 0, 0, 0.2, 0, -10, None, None, None, None, None, None, None, None, None]])
        }
        with self.assertRaises(BusSpecError):
            Simulator(network, 1, 100)

    def test_no_slack_dev(self):
        network = {
            'baseMVA': 100,
            'bus': np.array([[0, 0, 50, 1.1, 0.9]]),
            'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90]]),
            'device': np.array([
                [0, 0, -1, 0.2, 0, -10, None, None, None, None, None, None, None, None, None]])
        }
        with self.assertRaises(DeviceSpecError):
            Simulator(network, 1, 100)

    def test_too_many_slack_buses(self):
        network = {
            'baseMVA': 100,
            'bus': np.array([[0, 0, 50, 1.1, 0.9],
                             [1, 0, 50, 1.1, 0.9]]),
            'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90]]),
            'device': np.array([
                [0, 0, 0, 0.2, 0, -10, None, None, None, None, None, None, None, None, None]])
        }
        with self.assertRaises(BusSpecError):
            Simulator(network, 1, 100)

    def test_too_many_slack_devs(self):
        network = {
            'baseMVA': 100,
            'bus': np.array([[0, 0, 50, 1.1, 0.9],
                             [1, 1, 50, 1.1, 0.9]]),
            'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90]]),
            'device': np.array([
                [0, 0, 0, 0.2, 0, -10, None, None, None, None, None, None, None, None, None],
                [1, 1, 0, 0.2, 0, -10, None, None, None, None, None, None, None, None, None]])
        }
        with self.assertRaises(DeviceSpecError):
            Simulator(network, 1, 100)

    def test_slack_dev_not_at_slack_bus(self):
        network = {
            'baseMVA': 100,
            'bus': np.array([[0, 0, 50, 1.1, 0.9],
                             [1, 1, 50, 1.1, 0.9]]),
            'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90]]),
            'device': np.array([
                [0, 1, 0, 0.2, 0, -10, None, None, None, None, None, None, None, None, None]])
        }
        with self.assertRaises(DeviceSpecError):
            Simulator(network, 1, 100)


if __name__ == '__main__':
    unittest.main()
