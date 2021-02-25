import unittest
import os
import numpy as np

from gym_anm.simulator.components import Bus
from gym_anm.simulator.components.errors import *


class TestBus(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # Set the working directory to the root.

    def test_simple_bus(self):
        spec = np.array([2, 1, 10, 1.5, 1.])
        bus = Bus(spec)

        self.assertEqual(bus.id, spec[0])
        self.assertEqual(bus.type, spec[1])
        self.assertEqual(bus.baseKV, spec[2])
        self.assertEqual(bus.v_max, spec[3])
        self.assertEqual(bus.v_min, spec[4])
        self.assertEqual(bus.is_slack, False)

    def test_simple_slack(self):
        spec = np.array([2, 0, 10, 1.5, 1.])
        bus = Bus(spec)

        self.assertEqual(bus.id, spec[0])
        self.assertEqual(bus.type, spec[1])
        self.assertEqual(bus.baseKV, spec[2])
        self.assertEqual(bus.v_max, spec[3])
        self.assertEqual(bus.v_min, spec[4])
        self.assertEqual(bus.is_slack, True)

    def test_bad_type(self):
        spec = np.array([2, 2, 10, 1.5, 1.])

        with self.assertRaises(BusSpecError):
            bus = Bus(spec)

    def test_negative_baskV(self):
        spec = np.array([2, 1, -1, 1.5, 1.])

        with self.assertRaises(BusSpecError):
            bus = Bus(spec)

    def test_negative_v_bounds(self):
        spec = np.array([2, 1, 10, -1, 1.])
        with self.assertRaises(BusSpecError):
            bus = Bus(spec)

    def test_infeasible_v_bounds(self):
        spec = np.array([2, 1, 10, 0.5, 1])
        with self.assertRaises(BusSpecError):
            bus = Bus(spec)


if __name__ == '__main__':
    unittest.main()
