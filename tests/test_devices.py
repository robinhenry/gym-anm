import unittest
import numpy.testing as npt

from gym_smartgrid.simulator.components import Load, Storage, Generator


class TestDevices(unittest.TestCase):
    def setUp(self):
        load_case = [3., -1., 0.26, 0., 0., 1., 0., -10., 0., 0., 0., 0., 0., 0., 0., 0.]
        gen_case = [3, 3, 0.1, 5., -5., 1, 20, 0, 20, 30, -5, 5, -3, 1, 0, 0]
        slack_case = [0, 0, 0., 100, -100, 1, 100, -100, 0, 0, 0, 0, 0, 0, 0, 0]
        storage_case = [5, 4, 0, 3, -3, 1, 5, -5, 5, 7, -2.94, 2.94, -1.69,
                        1.69, 50, 0.9]

        self.load = Load(1, 1, load_case)
        self.gen = Generator(1, 1, gen_case)
        self.slack = Generator(0, 0, slack_case)
        self.storage = Storage(1, 2, storage_case)

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
        npt.assert_almost_equal(gen.lead_off, 13.)
        npt.assert_almost_equal(gen.lag_slope, 0.2)
        npt.assert_almost_equal(gen.lag_off, -9.)

    def test_load_compute_pq(self):
        ps = [-10, 0, 1, 10]
        for p in ps:
            self.load.compute_pq(p)
            self.assertEqual(self.load.q, p * 0.26)

    def test_gen_compute_pq(self):
        raise NotImplementedError

    def test_storage_compute_pq(self):
        raise NotImplementedError

    def test_storage_manage(self):
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()
