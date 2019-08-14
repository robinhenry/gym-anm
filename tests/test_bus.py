import unittest
import numpy.testing as npt

from gym_smartgrid.simulator.components import Bus

class TestBus(unittest.TestCase):

    def setUp(self):
        self.slack = Bus([0, 3, 0.5, 2., 10, 1.04, 1.04])
        self.pq = Bus([1, 1, 0.5, 2., 150, 1.1, 0.9])

    def test_slack(self):
        self.assertEqual(self.slack.id, 0)
        self.assertEqual(self.slack.type, 3)
        npt.assert_almost_equal(self.slack.v_max, 1.04)
        npt.assert_almost_equal(self.slack.v_min, 1.04)
        npt.assert_almost_equal(self.slack.baseKV, 10.)
        self.assertTrue(self.slack.is_slack)
        npt.assert_almost_equal(self.slack.v_slack, 1.04)

    def test_pq(self):
        self.assertEqual(self.pq.id, 1)
        self.assertEqual(self.pq.type, 1)
        npt.assert_almost_equal(self.pq.v_max, 1.1)
        npt.assert_almost_equal(self.pq.v_min, 0.9)
        npt.assert_almost_equal(self.pq.baseKV, 150.)
        self.assertFalse(self.pq.is_slack)

        with self.assertRaises(AttributeError):
            _ = self.pq.v_slack

if __name__ == '__main__':
    unittest.main()
