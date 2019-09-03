import unittest
import numpy.testing as npt
import numpy as np
import os

from gym_anm.simulator import Simulator


class TestPFE(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # Set the working directory to the root.
        self.d = 5

    def test_2bus(self):
        """ A distribution network with only 2 buses. """

        case = {'baseMVA': 100}
        case['bus'] = np.array([
            [0, 3, 10, 1, 1],
            [1, 1, 10, 1.1, 0.9]
        ])
        case['device'] = np.array([
            [0,  0, 0, 30,   -30, 1, 40,    0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 1, 100, -100, 1,  1, -100, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        case['branch'] = np.array([
            [0, 1, 1, 1, 0, 100, 0, 0, 1]
        ])

        simulator = Simulator(case)
        state, _, _, _ = simulator.transition([-10], [], [], [], [])

        npt.assert_almost_equal(state['P_DEV'], [13.81966011, -10.], decimal=self.d)
        npt.assert_almost_equal(state['Q_DEV'], [13.81966011, -10.], decimal=self.d)
        npt.assert_almost_equal(state['P_BUS'], [13.81966011, -10.], decimal=self.d)
        npt.assert_almost_equal(state['Q_BUS'], [13.81966011, -10.], decimal=self.d)
        npt.assert_almost_equal(state['V_MAGN_BUS'], [1, 0.7236067], decimal=self.d)
        npt.assert_almost_equal(state['P_BR_F'], [13.81966011], decimal=self.d)
        npt.assert_almost_equal(state['P_BR_T'], [-10], decimal=self.d)
        npt.assert_almost_equal(state['Q_BR_F'], [13.81966011], decimal=self.d)
        npt.assert_almost_equal(state['Q_BR_T'], [-10], decimal=self.d)


    def test_3bus(self):
        """ A radial distribution network with 3 buses. """

        case = {'baseMVA': 100}
        case['bus'] = np.array([
            [0, 3, 10, 1, 1],
            [1, 1, 10, 1.1, 0.9],
            [2, 1, 10, 1.1, 0.9]
        ])
        case['device'] = np.array([
            [0, 0, 0, 50, -50, 1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 1, 100, -100, 1, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, -1, 1, 100, -100, 1, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        case['branch'] = np.array([
            [0, 1,  1,   1, .1, 100, 0, 0, 1],
            [0, 2, .5, .25, .3, 100, 0, 0, 1]
        ])

        simulator = Simulator(case)
        state, _, _, _ = simulator.transition([-10, -5], [], [], [], [])

        npt.assert_almost_equal(state['P_DEV'], [18.03733707, -10., -5.], decimal=self.d)
        npt.assert_almost_equal(state['Q_DEV'], [-20.27524390, -10., -5.], decimal=self.d)
        npt.assert_almost_equal(state['P_BUS'], [18.03733707, -10., -5.], decimal=self.d)
        npt.assert_almost_equal(state['Q_BUS'], [-20.27524390, -10., -5.], decimal=self.d)
        npt.assert_almost_equal(state['P_BR_F'], [12.41600605, 5.62133101], decimal=self.d)
        npt.assert_almost_equal(state['P_BR_T'], [-10, -5], decimal=self.d)
        npt.assert_almost_equal(state['Q_BR_F'], [4.351117845091385, -24.62636175], decimal=self.d)
        npt.assert_almost_equal(state['Q_BR_T'], [-10, -5], decimal=self.d)
        npt.assert_almost_equal(state['V_MAGN_BUS'], [1, 0.78292888, 0.99789870], decimal=self.d)

    def test_3bus_loop(self):
        """
        A 3-bus loop distribution network:

        Gen -- Load
         |      |
        Load ----
        """

        case = {'baseMVA': 100}
        case['bus'] = np.array([
            [0, 3, 10, 1, 1],
            [1, 1, 10, 1.1, 0.9],
            [2, 1, 10, 1.1, 0.9]
        ])
        case['device'] = np.array([
            [0, 0, 0, 50, -50, 1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 1, 100, -100, 1, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, -1, 1, 100, -100, 1, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        case['branch'] = np.array([
            [0, 1,  1,   1, 0, 100, 0, 0, 1],
            [1, 2, .5, .25, 0, 100, 0, 0, 1],
            [0, 2, .75, .3, 0, 100, 0, 0, 1]
        ])

        simulator = Simulator(case)
        state, _, _, _ = simulator.transition([-10, -5], [], [], [], [])

        npt.assert_almost_equal(state['P_DEV'], [17.87248775, -10, -5], decimal=self.d)
        npt.assert_almost_equal(state['Q_DEV'], [16.85076239, -10, -5], decimal=self.d)
        npt.assert_almost_equal(state['P_BUS'], [17.87248775, -10, -5], decimal=self.d)
        npt.assert_almost_equal(state['Q_BUS'], [16.85076239, -10, -5], decimal=self.d)
        npt.assert_almost_equal(state['V_MAGN_BUS'], [1, 0.85183036, 0.88814474], decimal=self.d)
        npt.assert_almost_equal(state['P_BR_F'], [6.17616917, -4.95827630, 11.69631857], decimal=self.d)
        npt.assert_almost_equal(state['P_BR_T'], [-5.04172369, 5.16927629, -10.16927629], decimal=self.d)
        npt.assert_almost_equal(state['Q_BR_F'], [8.67752747, -2.45691801, 8.173234915], decimal=self.d)
        npt.assert_almost_equal(state['Q_BR_T'], [-7.54308198, 2.56241800, -7.562418004], decimal=self.d)

if __name__ == '__main__':
    unittest.main()