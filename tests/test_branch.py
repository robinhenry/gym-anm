import unittest
import numpy.testing as npt

from gym_smartgrid.simulator.components import TransmissionLine

class TestBranch(unittest.TestCase):

    def setUp(self):
        self.branch = \
            TransmissionLine([0, 1, 2., 2., 1.5, 5, 0., 0., 1, 0., 360.], 100)
        self.transformer = \
            TransmissionLine([0, 1, 2., 2., 1.5, 5, 2., 30, 1, 0., 360.], 100)

    def test_branch(self):
        self.assertEqual(self.branch.f_bus, 0)
        self.assertEqual(self.branch.t_bus, 1)
        npt.assert_almost_equal(self.branch.ang_min, 0.)
        npt.assert_almost_equal(self.branch.ang_max, 360.)
        npt.assert_almost_equal(self.branch.i_max, 0.05)

        npt.assert_almost_equal(self.branch.tap, 1.)

        npt.assert_almost_equal(self.branch.series.real, 0.25)
        npt.assert_almost_equal(self.branch.series.imag, -0.25)

        npt.assert_almost_equal(self.branch.shunt.real, 0.)
        npt.assert_almost_equal(self.branch.shunt.imag,0.75)

    def test_transformer(self):
        npt.assert_almost_equal(self.transformer.tap.real, 1.73205, decimal=5)
        npt.assert_almost_equal(self.transformer.tap.imag, 0.99999, decimal=5)


if __name__ == '__main__':
    unittest.main()
