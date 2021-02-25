import unittest
import os
import numpy as np

from gym_anm.simulator.components import Load, StorageUnit, Generator, Device
from gym_anm.simulator.components.errors import *


class TestDevice(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # Set the working directory to the root.

        self.bus_ids = [0, 1, 2]
        self.baseMVA = 10

    def test_bad_bus_id(self):
        for b in [None, -1, 3, 4]:
            spec = np.array([2, b, 1, None, None, None, None, None, None, None, None, None, None, None, None])
            with self.assertRaises(DeviceSpecError):
                Device(spec, self.bus_ids, self.baseMVA)

    def test_bad_type(self):
        for t in [None, -3, -2, 4, 5]:
            spec = np.array([2, 3, t, None, None, None, None, None, None, None, None, None, None, None, None])
            with self.assertRaises(DeviceSpecError):
                Device(spec, self.bus_ids, self.baseMVA)


class TestLoad(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # Set the working directory to the root.

        self.bus_ids = [0, 1, 2]
        self.baseMVA = 10

    def test_simple_load(self):
        spec = np.array([2, 1, -1, 0.2, 0, -10, None, None, None, None, None, None, None, None, None])
        load = Load(spec, self.bus_ids, self.baseMVA)

        self.assertEqual(load.dev_id, spec[0])
        self.assertEqual(load.bus_id, spec[1])
        self.assertEqual(load.type, spec[2])
        self.assertEqual(load.qp_ratio, spec[3])
        self.assertEqual(load.p_max, spec[4] / self.baseMVA)
        self.assertEqual(load.p_min, spec[5] / self.baseMVA)
        self.assertEqual(load.q_max, spec[3] * spec[4] / self.baseMVA)
        self.assertEqual(load.q_min, spec[3] * spec[5] / self.baseMVA)
        self.assertIsNone(load.p_plus)
        self.assertIsNone(load.q_minus)
        self.assertIsNone(load.soc_max)
        self.assertIsNone(load.soc_min)
        self.assertIsNone(load.eff)
        self.assertFalse(load.is_slack)

    def test_slack_load(self):
        spec = np.array([2, 1, 0, 0.2, 0, -10, None, None, None, None, None, None, None, None, None])
        with self.assertRaises(LoadSpecError):
            Load(spec, self.bus_ids, self.baseMVA)

    def test_positive_p_max(self):
        spec = np.array([2, 1, -1, 0.2, 1, -10, None, None, None, None, None, None, None, None, None])
        with self.assertRaises(LoadSpecError):
            Load(spec, self.bus_ids, self.baseMVA)

    def test_infeasible_p_bounds(self):
        spec = np.array([2, 1, -1, 0.2, -5, -2, None, None, None, None, None, None, None, None, None])
        with self.assertRaises(LoadSpecError):
            Load(spec, self.bus_ids, self.baseMVA)

    def test_map_pq(self):
        spec = np.array([2, 1, -1, 0.2, 0, -10, None, None, None, None, None, None, None, None, None])
        load = Load(spec, self.bus_ids, self.baseMVA)

        # p_min < p < p_max
        for p in np.random.uniform(spec[5], spec[4], 50):
            p /= self.baseMVA
            load.map_pq(p)
            self.assertEqual(load.p, p)
            self.assertEqual(load.q, p * spec[3])

        # p < p_min
        for p in np.random.uniform(5*spec[5], spec[5], 50):
            p /= self.baseMVA
            load.map_pq(p)
            self.assertEqual(load.p, spec[5] / self.baseMVA)
            self.assertEqual(load.q, spec[5] * spec[3] / self.baseMVA)

        # p > 0
        for p in np.random.uniform(0.5, 50, 50):
            p /= self.baseMVA
            load.map_pq(p)
            self.assertEqual(load.p, 0)
            self.assertEqual(load.q, 0)


class TestGenerator(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # Set the working directory to the root.

        self.bus_ids = [0, 1, 2]
        self.baseMVA = 10

    def test_simple_gen(self):
        spec = np.array([2, 1, 1, None, 10, 1, 10, -10, 5, None, 5, -5, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)

        self.assertEqual(gen.dev_id, spec[0])
        self.assertEqual(gen.bus_id, spec[1])
        self.assertEqual(gen.type, spec[2])
        self.assertIsNone(gen.qp_ratio)
        self.assertEqual(gen.p_max, spec[4] / self.baseMVA)
        self.assertEqual(gen.p_min, spec[5] / self.baseMVA)
        self.assertEqual(gen.q_max, spec[6] / self.baseMVA)
        self.assertEqual(gen.q_min, spec[7] / self.baseMVA)
        self.assertEqual(gen.p_plus, spec[8] / self.baseMVA)
        self.assertEqual(gen.p_minus, None)
        self.assertEqual(gen.q_plus, spec[10] / self.baseMVA)
        self.assertEqual(gen.q_minus, spec[11] / self.baseMVA)
        self.assertIsNone(gen.soc_max)
        self.assertIsNone(gen.soc_min)
        self.assertIsNone(gen.eff)
        self.assertFalse(gen.is_slack)

        tau_1 = (spec[10] - spec[6]) / (spec[4] - spec[8])
        tau_2 = (spec[11] - spec[7]) / (spec[4] - spec[8])
        self.assertAlmostEqual(gen.tau_1, tau_1, places=5)
        self.assertAlmostEqual(gen.tau_2, tau_2, places=5)
        self.assertIsNone(gen.tau_3)
        self.assertIsNone(gen.tau_4)

        rho_1 = (spec[6] - tau_1 * spec[8]) / self.baseMVA
        rho_2 = (spec[7] - tau_2 * spec[8]) / self.baseMVA
        self.assertAlmostEqual(gen.rho_1, rho_1, places=5)
        self.assertAlmostEqual(gen.rho_2, rho_2, places=5)
        self.assertIsNone(gen.rho_3)
        self.assertIsNone(gen.rho_4)

    def test_slack_gen(self):
        spec = np.array([2, 1, 0, None, 10, 1, 10, -10, 5, None, None, None, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)

        self.assertTrue(gen.is_slack)

    def test_negative_p_min(self):
        spec = np.array([2, 1, 1, None, 10, -1, 10, -10, None, None, None, None, None, None, None])
        with self.assertRaises(GenSpecError):
            Generator(spec, self.bus_ids, self.baseMVA)

    def test_infeasible_p_bounds(self):
        spec = np.array([2, 1, 1, None, 1, 2, 10, -10, None, None, None, None, None, None, None])
        with self.assertRaises(GenSpecError):
            Generator(spec, self.bus_ids, self.baseMVA)

    def test_infeasible_q_bounds(self):
        spec = np.array([2, 1, 1, None, 10, 1, 1, 2, None, None, None, None, None, None, None])
        with self.assertRaises(GenSpecError):
            Generator(spec, self.bus_ids, self.baseMVA)

    def test_defaut_p_values(self):
        # p_max = None
        spec = np.array([2, 1, 1, None, None, 1, 10, 1, None, None, None, None, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(gen.p_max, np.inf)

        # p_max = 0
        spec = np.array([2, 1, 1, None, 0, 0, 10, 1, None, None, None, None, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(gen.p_max, 0)

        # p_min = None
        spec = np.array([2, 1, 1, None, 10, None, 10, 1, None, None, None, None, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(gen.p_min, 0)

    def test_default_q_values(self):
        # q_max = None
        spec = np.array([2, 1, 1, None, 10, 1, None, 1, None, None, None, None, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(gen.q_max, np.inf)

        # q_min = None
        spec = np.array([2, 1, 1, None, 10, 1, 10, None, None, None, None, None, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(gen.q_min, -np.inf)

    def test_p_plus(self):
        # p_plus < p_min and p_plus > p_max
        for p_plus in [0.5, 11]:
            spec = np.array([2, 1, 1, None, 10, 1, 10, 1, p_plus, None, None, None, None, None, None])
            with self.assertRaises(GenSpecError):
                Generator(spec, self.bus_ids, self.baseMVA)

        # p_plus = None
        spec = np.array([2, 1, 1, None, 10, 1, 10, 1, None, None, None, None, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(gen.p_plus, spec[4] / self.baseMVA)

    def test_p_minus(self):
        for p_minus in [None, 0, -1, 1]:
            spec = np.array([2, 1, 1, None, 10, 1, 10, 1, None, p_minus, None, None, None, None, None])
            gen = Generator(spec, self.bus_ids, self.baseMVA)
            self.assertIsNone(gen.p_minus)

    def test_q_plus(self):
        # q_plus < q_min and q_plus > q_max
        for q_plus in [0.5, 11]:
            spec = np.array([2, 1, 1, None, 10, 1, 10, 1, None, None, q_plus, None, None, None, None])
            with self.assertRaises(GenSpecError):
                Generator(spec, self.bus_ids, self.baseMVA)

        # q_plus = None
        spec = np.array([2, 1, 1, None, 10, 1, 15, 1, None, None, None, None, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(gen.q_plus, spec[6] / self.baseMVA)

    def test_q_minus(self):
        # q_minus < q_min and q_minus > q_max
        for q_minus in [0.5, 11]:
            spec = np.array([2, 1, 1, None, 10, 1, 10, 1, None, None, None, q_minus, None, None, None])
            with self.assertRaises(GenSpecError):
                Generator(spec, self.bus_ids, self.baseMVA)

        # q_minus = None
        spec = np.array([2, 1, 1, None, 10, 1, 15, 1, None, None, None, None, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(gen.q_minus, spec[7] / self.baseMVA)

    def test_infeasible_q_plus_minus(self):
        # q_minus > q_plus
        spec = np.array([2, 1, 1, None, 10, 1, 15, 1, None, None, 1, 2, None, None, None])
        with self.assertRaises(GenSpecError):
            Generator(spec, self.bus_ids, self.baseMVA)

    def test_setting_infeasible_ppot(self):
        spec = np.array([2, 1, 1, None, 10, 1, 2, -3, None, None, None, None, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)

        for p_pot in np.random.uniform(-10, 10, 50) / self.baseMVA:
            gen.p_pot = p_pot
            true = np.clip(p_pot, spec[5] / self.baseMVA, spec[4] / self.baseMVA)
            self.assertAlmostEqual(gen.p_pot, float(true), places=5)

    def test_no_flexibility_limits(self):
        p_max = 10
        for p_plus in [None, p_max]:
            spec = np.array([2, 1, 1, None, p_max, 1, 2, -3, p_plus, None, 1, -1, None, None, None])
            gen = Generator(spec, self.bus_ids, self.baseMVA)
            self.assertEqual(gen.tau_1, 0)
            self.assertEqual(gen.tau_2, 0)

        q_max = 5
        for q_plus in [None, q_max]:
            spec = np.array([2, 1, 1, None, 10, 1, q_max, -3, 5, None, q_plus, -1, None, None, None])
            gen = Generator(spec, self.bus_ids, self.baseMVA)
            self.assertEqual(gen.tau_1, 0)

        q_min = -3
        for q_minus in [None, q_min]:
            spec = np.array([2, 1, 1, None, 10, 1, 2, q_min, 5, None, 1, q_minus, None, None, None])
            gen = Generator(spec, self.bus_ids, self.baseMVA)
            self.assertEqual(gen.tau_2, 0)

    def test_map_pq_no_flexibility_limits(self):
        spec = np.array([2, 1, 1, None, 10, 1, 2, -3, None, None, None, None, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)

        ps = np.random.uniform(-10, 10, 10) / self.baseMVA
        qs = np.random.uniform(-10, 10, 10) / self.baseMVA
        p_pots = np.random.uniform(spec[5], spec[4], 10) / self.baseMVA
        for p in ps:
            for q in qs:
                for p_pot in p_pots:
                    gen.p_pot = p_pot
                    gen.map_pq(p, q)
                    true_p = np.clip(p, spec[5] / self.baseMVA,
                                     np.minimum(spec[4] / self.baseMVA, p_pot))
                    true_q = np.clip(q, spec[7] / self.baseMVA,
                                     spec[6] / self.baseMVA)
                    self.assertAlmostEqual(gen.p, true_p, places=5)
                    self.assertAlmostEqual(gen.q, true_q, places=5)

    def test_map_pq_with_flex_limits(self):
        spec = np.array([2, 1, 1, None, 10, 1, 2, -2, 9, None, 1, -1, None, None, None])
        gen = Generator(spec, self.bus_ids, self.baseMVA)

        points = np.array([(-1, 0.5), (5, 5), (5, -5), (12, 0), (10, 2), (10, -2)]) / self.baseMVA
        mapped = np.array([(1, 0.5), (5, 2), (5, -2), (10, 0), (9.5, 1.5), (9.5, -1.5)]) / self.baseMVA
        for p, m in zip(points, mapped):
            gen.map_pq(p[0], p[1])
            self.assertAlmostEqual(gen.p, m[0], places=5)
            self.assertAlmostEqual(gen.q, m[1], places=5)


class TestStorageUnit(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))

        self.bus_ids = [0, 1, 2]
        self.baseMVA = 10

    def test_simple_storage(self):
        spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, -6, 10, -15, 100, 10, 0.9])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)

        self.assertEqual(su.dev_id, spec[0])
        self.assertEqual(su.bus_id, spec[1])
        self.assertEqual(su.type, spec[2])
        self.assertIsNone(su.qp_ratio)
        self.assertEqual(su.p_max, spec[4] / self.baseMVA)
        self.assertEqual(su.p_min, spec[5] / self.baseMVA)
        self.assertEqual(su.q_max, spec[6] / self.baseMVA)
        self.assertEqual(su.q_min, spec[7] / self.baseMVA)
        self.assertEqual(su.p_plus, spec[8] / self.baseMVA)
        self.assertEqual(su.p_minus, spec[9] / self.baseMVA)
        self.assertEqual(su.q_plus, spec[10] / self.baseMVA)
        self.assertEqual(su.q_minus, spec[11] / self.baseMVA)
        self.assertEqual(su.soc_max, spec[12] / self.baseMVA)
        self.assertEqual(su.soc_min, spec[13] / self.baseMVA)
        self.assertEqual(su.eff, spec[14])
        self.assertFalse(su.is_slack)

        tau_1 = (spec[10] - spec[6]) / (spec[4] - spec[8])
        tau_2 = (spec[11] - spec[7]) / (spec[4] - spec[8])
        tau_3 = (spec[7] - spec[11]) / (spec[9] - spec[5])
        tau_4 = (spec[6] - spec[10]) / (spec[9] - spec[5])
        self.assertAlmostEqual(su.tau_1, tau_1, places=5)
        self.assertAlmostEqual(su.tau_2, tau_2, places=5)
        self.assertAlmostEqual(su.tau_3, tau_3, places=5)
        self.assertAlmostEqual(su.tau_4, tau_4, places=5)

        rho_1 = (spec[6] - tau_1 * spec[8]) / self.baseMVA
        rho_2 = (spec[7] - tau_2 * spec[8]) / self.baseMVA
        rho_3 = (spec[7] - tau_3 * spec[9]) / self.baseMVA
        rho_4 = (spec[6] - tau_4 * spec[9]) / self.baseMVA
        self.assertAlmostEqual(su.rho_1, rho_1, places=5)
        self.assertAlmostEqual(su.rho_2, rho_2, places=5)
        self.assertAlmostEqual(su.rho_3, rho_3, places=5)
        self.assertAlmostEqual(su.rho_4, rho_4, places=5)

    def test_infeasible_p_bounds(self):
        spec = np.array([2, 1, 3, None, 10, 12, 20, -30, 5, -6, 10, -15, 100, 10, 0.9])
        with self.assertRaises(StorageSpecError):
            StorageUnit(spec, self.bus_ids, self.baseMVA)


    def test_infeasible_q_bounds(self):
        spec = np.array([2, 1, 3, None, 10, -12, 20, 30, 5, -6, 10, -15, 100, 10, 0.9])
        with self.assertRaises(StorageSpecError):
            StorageUnit(spec, self.bus_ids, self.baseMVA)

    def test_defaut_p_values(self):
        # p_max = None
        spec = np.array([2, 1, 3, None, None, -12, 20, -30, 5, -6, 10, -15, 100, 10, 0.9])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(su.p_max, np.inf)

        # p_max < 0
        spec = np.array([2, 1, 3, None, -10, -12, 20, -30, 5, -6, 10, -15, 100, 10, 0.9])
        with self.assertRaises(StorageSpecError):
            StorageUnit(spec, self.bus_ids, self.baseMVA)

        # p_min = None
        spec = np.array([2, 1, 3, None, 10, None, 20, -30, 5, -6, 10, -15, 100, 10, 0.9])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(su.p_min, -np.inf)

        # p_min >= 0
        spec = np.array([2, 1, 3, None, 10, 5, 20, -30, 5, -6, 10, -15, 100, 10, 0.9])
        with self.assertRaises(StorageSpecError):
            StorageUnit(spec, self.bus_ids, self.baseMVA)

    def test_default_q_values(self):
        # q_max = None
        spec = np.array([2, 1, 3, None, 10, -12, None, -30, 5, -6, 10, -15, 100, 10, 0.9])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(su.q_max, np.inf)

        # q_min = None
        spec = np.array([2, 1, 3, None, 10, -12, 20, None, 5, -6, 10, -15, 100, 10, 0.9])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(su.q_min, -np.inf)

    def test_p_plus(self):
        # p_plus < p_min and p_plus > p_max
        for p_plus in [-15, 25]:
            spec = np.array([2, 1, 3, None, 10, -12, 20, -30, p_plus, -6, 10, -15, 100, 10, 0.9])
            with self.assertRaises(StorageSpecError):
                StorageUnit(spec, self.bus_ids, self.baseMVA)

        # p_plus = None
        spec = np.array([2, 1, 3, None, 10, -12, 20, -30, None, -6, 10, -15, 100, 10, 0.9])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(su.p_plus, spec[4] / self.baseMVA)

    def test_p_minus(self):
        # p_minus < p_min and p_minus > p_max
        for p_minus in [-15, 25]:
            spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, p_minus, 10, -15, 100, 10, 0.9])
            with self.assertRaises(StorageSpecError):
                StorageUnit(spec, self.bus_ids, self.baseMVA)

        # p_minus = None
        spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, None, 10, -15, 100, 10, 0.9])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(su.p_minus, spec[5] / self.baseMVA)

    def test_q_plus(self):
       # q_plus < q_min and q_plus > q_max
        for q_plus in [-35, 25]:
            spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, -6, q_plus, -15, 100, 10, 0.9])
            with self.assertRaises(StorageSpecError):
                StorageUnit(spec, self.bus_ids, self.baseMVA)

        # q_plus = None
        spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, -6, None, -15, 100, 10, 0.9])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(su.q_plus, spec[6] / self.baseMVA)

    def test_q_minus(self):
        # q_minus < q_min and q_minus > q_max
        for q_minus in [-35, 25]:
            spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, -6, 10, q_minus, 100, 10, 0.9])
            with self.assertRaises(StorageSpecError):
                StorageUnit(spec, self.bus_ids, self.baseMVA)

        # q_minus = None
        spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, -6, 10, None, 100, 10, 0.9])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)
        self.assertEqual(su.q_minus, spec[7] / self.baseMVA)

    def test_infeasible_q_plus_minus(self):
        # q_minus > q_plus
        spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, -6, 10, 15, 100, 10, 0.9])
        with self.assertRaises(StorageSpecError):
            StorageUnit(spec, self.bus_ids, self.baseMVA)

    def test_negative_soc_bounds(self):
        spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, -6, 10, -15, 100, -1, 0.9])
        with self.assertRaises(StorageSpecError):
            StorageUnit(spec, self.bus_ids, self.baseMVA)

    def test_infeasible_soc_bounds(self):
        spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, -6, 10, -15, 10, 20, 0.9])
        with self.assertRaises(StorageSpecError):
            StorageUnit(spec, self.bus_ids, self.baseMVA)

    def test_bad_eff(self):
        for eff in np.concatenate((np.random.uniform(-1, 0, 10), np.random.uniform(1.0001, 10, 10))):
            spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, -6, 10, -15, 100, 10, eff])
            with self.assertRaises(StorageSpecError):
                StorageUnit(spec, self.bus_ids, self.baseMVA)

    def test_update_soc_simple(self):
        spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, -6, 10, -15, 100, 10, 0.9])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)

        # p > 0
        for p in np.random.uniform(0, 10, 50):
            for delta_t in [1., 0.25]:
                su.soc = 50 / self.baseMVA
                su.p = p / self.baseMVA
                su.update_soc(delta_t)
                self.assertAlmostEqual(su.soc * self.baseMVA, 50 - delta_t * p / spec[14],
                                       places=5)

        # p < 0
        for p in np.random.uniform(-10, 0, 50):
            for delta_t in [1., 0.25]:
                su.soc = 50 / self.baseMVA
                su.p = p / self.baseMVA
                su.update_soc(delta_t)
                self.assertAlmostEqual(su.soc * self.baseMVA, 50 - delta_t * p * spec[14],
                                       places=5)

    def test_update_soc_clipping(self):
        spec = np.array([2, 1, 3, None, 10, -12, 20, -30, 5, -6, 10, -15, 10, 0, 1])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)

        # p < 0
        for p in np.random.uniform(spec[5], -1, 50):
            su.soc = 9 / self.baseMVA
            su.p = p / self.baseMVA
            su.update_soc(delta_t=1)
            self.assertEqual(su.soc, spec[12] / self.baseMVA)

        # p > 0
        for p in np.random.uniform(1, spec[4], 50):
            su.soc = 1 / self.baseMVA
            su.p = p / self.baseMVA
            su.update_soc(delta_t=1)
            self.assertEqual(su.soc, spec[13] / self.baseMVA)

    def test_no_flexibility_limits(self):
        p_max = 10
        for p_plus in [None, p_max]:
            spec = np.array([2, 1, 3, None, p_max, -12, 20, -30, p_plus, -6, 10, -15, 10, 0, 1])
            su = StorageUnit(spec, self.bus_ids, self.baseMVA)
            self.assertEqual(su.tau_1, 0)
            self.assertEqual(su.tau_2, 0)

        p_min = - 12
        for p_minus in [None, p_min]:
            spec = np.array([2, 1, 3, None, 10, p_min, 20, -30, 5, p_minus, 10, -15, 10, 0, 1])
            su = StorageUnit(spec, self.bus_ids, self.baseMVA)
            self.assertEqual(su.tau_3, 0)
            self.assertEqual(su.tau_4, 0)

        q_max = 20
        for q_plus in [None, q_max]:
            spec = np.array([2, 1, 3, None, 10, -12, q_max, -30, 5, -6, q_plus, -15, 10, 0, 1])
            su = StorageUnit(spec, self.bus_ids, self.baseMVA)
            self.assertEqual(su.tau_1, 0)
            self.assertEqual(su.tau_4, 0)

        q_min = -30
        for q_minus in [None, q_min]:
            spec = np.array([2, 1, 3, None, 10, -12, 20, q_min, 5, -6, 10, q_minus, 10, 0, 1])
            su = StorageUnit(spec, self.bus_ids, self.baseMVA)
            self.assertEqual(su.tau_2, 0)
            self.assertEqual(su.tau_3, 0)

    def test_map_pq_no_flexibility_limits(self):
        spec = np.array([2, 1, 3, None, 10, -12, 20, -30, None, None, None, None, 1000, 0, 1])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)
        delta_t = 1.

        ps = list(np.random.uniform(-20, -12, 10) / self.baseMVA) + list(np.random.uniform(10.01, 20, 10) / self.baseMVA)
        qs = list(np.random.uniform(-40, -30, 10) / self.baseMVA) + list(np.random.uniform(20.01, 30, 10) / self.baseMVA)

        for p in ps:
            for q in qs:
                su.soc = spec[12] / self.baseMVA / 2
                su.map_pq(p, q, delta_t)
                true_p = np.clip(p, spec[5] / self.baseMVA, spec[4] / self.baseMVA)
                true_q = np.clip(q, spec[7] / self.baseMVA, spec[6] / self.baseMVA)
                self.assertEqual(su.p, true_p)
                self.assertEqual(su.q, true_q)

        for p in np.array([-10, -5, 0, 5, 10]) / self.baseMVA:
            for q in np.array([-30, -10, 0, 15]) / self.baseMVA:
                su.soc = spec[12] / self.baseMVA / 2
                su.map_pq(p, q, delta_t)
                self.assertEqual(su.p, p)
                self.assertEqual(su.q, q)

    def test_map_pq_with_flex_limits(self):
        spec = np.array([2, 1, 3, None, 10, -11, 20, -30, 5, -6, 15, -25, 1000, 0, 1])
        su = StorageUnit(spec, self.bus_ids, self.baseMVA)
        delta_t = 1.

        points = np.array([(8.5, 18.5), (8.5, -28.5), (-9.5, 18.5), (-9.5, -28.5)]) / self.baseMVA
        mapped = np.array([(7.5, 17.5), (7.5, -27.5), (-8.5, 17.5), (-8.5, -27.5)]) / self.baseMVA
        for p, m in zip(points, mapped):
            su.soc = spec[12] / self.baseMVA / 2
            su.map_pq(p[0], p[1], delta_t)
            self.assertAlmostEqual(su.p, m[0])
            self.assertAlmostEqual(su.q, m[1])


if __name__ == '__main__':
    unittest.main()
