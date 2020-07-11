import unittest
import numpy.testing as npt
import os

from gym_anm.simulator.components import Load, StorageUnit, Generator


class TestDevices(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # Set the working directory to the root.

        load_case = [3., -1., .26, 0., 0., 1., 0., -10., 0., 0., 0., 0., 0., 0., 0., 0.]
        gen_case = [3, 3, .1, 5, -5, 1, 20, 0, 10, 20, -5, 5, -3, 1, 0, 0]
        slack_case = [0, 0, 0., 100, -100, 1, 100, -100, 0, 0, 0, 0, 0, 0, 0, 0]
        storage_case = [5, 4, 0, 5, -5, 1, 20, -20, 10, 20, -5, 5, -3, 1, 50, .9]

        self.load = Load(1, 1, )
        self.gen = Generator(1, 1, )
        self.slack = Generator(0, 0, )
        self.storage = StorageUnit(1, 2, )

    def test_init_load(self):
        self.assertEqual(self.load.dev_id, 1)
        self.assertEqual(self.load.type_id, 1)
        self.assertEqual(self.load.bus_id, 3)
        self.assertEqual(self.load.type, -1)
        self.assertFalse(self.load.is_slack)
        self.assertIsNone(self.load.soc_max)
        self.assertIsNone(self.load.soc_min)
        self.assertIsNone(self.load.eff)

        npt.assert_almost_equal(self.load.qp_ratio, 0.26)
        npt.assert_almost_equal(self.load.p_min, -10.)
        npt.assert_almost_equal(self.load.p_max, 0.)

    def test_init_gen(self):
        self.assertEqual(self.gen.dev_id, 1)
        self.assertEqual(self.gen.type_id, 1)
        self.assertEqual(self.gen.bus_id, 3)
        self.assertEqual(self.gen.type, 3)
        self.assertFalse(self.gen.is_slack)
        self.assertIsNone(self.gen.soc_max)
        self.assertIsNone(self.gen.soc_min)
        self.assertIsNone(self.gen.eff)

        npt.assert_almost_equal(self.gen.qp_ratio, 0.1)
        npt.assert_almost_equal(self.gen.q_max, 5)
        npt.assert_almost_equal(self.gen.q_min, -5)
        npt.assert_almost_equal(self.gen.p_max, 20)
        npt.assert_almost_equal(self.gen.p_min, 0.)

    def test_init_slack(self):
        self.assertTrue(self.slack.is_slack)

    def test_init_lag_lead_limits(self):
        gen = self.gen
        npt.assert_almost_equal(gen.lead_slope, -0.4)
        npt.assert_almost_equal(gen.lead_off, 9)
        npt.assert_almost_equal(gen.lag_slope, 0.2)
        npt.assert_almost_equal(gen.lag_off, -7)

    def test_load_compute_pq(self):
        ps = [-10, 0, 1, 10]
        for p in ps:
            self.load.map_pq(p)
            self.assertEqual(self.load.q, p * 0.26)

    def test_gen_compute_pq(self):
        gen = self.gen
        ps = [0, 5, 10, 15, 20, 30]
        p_desired = [0, 5, 10, 15, 18,  18]
        q_desired = [0, .5, 1, 1.5, 1.8, 1.8]

        for i, p in enumerate(ps):
            gen.map_pq(p)
            npt.assert_almost_equal(gen.p, p_desired[i])
            npt.assert_almost_equal(gen.q, q_desired[i])

    def test_storage_compute_pq(self):
        su = self.storage
        pq = [(5, 2), (5, -3), (5, 7), (5, -7), (15, 5), (15, -5), (22, 1), (22, -1)]
        desired = [(5, 2), (5, -3), (5, 5), (5, -5), (15, 3), (15, -4), (20, 1), (20, -1)]

        for point, des in zip(pq, desired):
            su.map_pq(*point)
            npt.assert_almost_equal(su.p, des[0])
            npt.assert_almost_equal(su.q, des[1])

            su.map_pq(- point[0], point[1])
            npt.assert_almost_equal(- su.p, des[0])
            npt.assert_almost_equal(su.q, des[1])

    def test_storage_manage_within_constraints(self):
        su = self.storage
        su.soc = 20
        delta_t = 0.25

        # Charging.
        su.manage(20, delta_t, 1)
        npt.assert_almost_equal(su.p, -20)
        npt.assert_almost_equal(su.q, 1)
        npt.assert_almost_equal(su.soc, 24.75)

        # Discharging.
        su.manage(-20, delta_t, -1)
        npt.assert_almost_equal(su.p, 18.94736842)
        npt.assert_almost_equal(su.q, -1)
        npt.assert_almost_equal(su.soc, 19.75)

    def test_storage_empty(self):
        su = self.storage
        su.soc = 1
        delta_t = 0.25

        su.manage(-10, delta_t, 0)
        npt.assert_almost_equal(su.p, 3.789473684)
        npt.assert_almost_equal(su.soc, 0)

    def test_storage_full(self):
        su = self.storage
        su.soc = 49
        delta_t = 0.25

        su.manage(10, delta_t, 0)
        npt.assert_almost_equal(su.p, -4.21052631)
        npt.assert_almost_equal(su.soc, 50)


if __name__ == '__main__':
    unittest.main()
