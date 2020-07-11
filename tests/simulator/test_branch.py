import numpy as np
import unittest
import numpy.testing as npt
import os

from gym_anm.simulator.components import TransmissionLine

class TestBranch(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # Set the working directory to the root.

        self.baseMVA = 100
        self.bus_ids = [1, 2]

        # No transformer
        br = np.array([1, 2, 2., 2., 1.5, 5., 1., 0.])
        self.branch = \
            TransmissionLine(br, self.baseMVA, self.bus_ids)

        # Transformer
        tr = np.array([2, 1, 2., 2., 1.5, 5, 2., 30,])
        self.transformer = \
            TransmissionLine(tr, self.baseMVA, self.bus_ids)

    def test_branch(self):
        self.assertEqual(self.branch.f_bus, 1)
        self.assertEqual(self.branch.t_bus, 2)
        npt.assert_almost_equal(self.branch.rate, 0.05)

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
