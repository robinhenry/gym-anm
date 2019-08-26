import unittest
import numpy.testing as npt
import numpy as np

from gym_smartgrid.simulator import Simulator


class TestSimulatorBus2OnlyLoad(unittest.TestCase):
    def setUp(self):
        case = {'baseMVA': 100}
        case['bus'] = np.array([
            [0, 3, 0, 0, 10, 1, 1],
            [1, 1, 0, 0, 10, 1.1, 0.9]
        ])
        case['device'] = np.array([
            [0,  0, 0, 30,   -30, 1, 40,    0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 1, 100, -100, 1,  1, -100, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        case['branch'] = np.array([
            [0, 1, 1, 1, 0, 100, 0, 0, 1, -360, 360]
        ])

        self.simulator = Simulator(case)

    def test_pfe(self):
        state, _, _, _ = self.simulator.transition([-10], [], [], [], [])

        npt.assert_almost_equal(state['P_DEV'], [13.81966011, -10.])
        npt.assert_almost_equal(state['Q_DEV'], [13.81966011, -10.])
        npt.assert_almost_equal(state['P_BUS'], [13.81966011, -10.])
        npt.assert_almost_equal(state['Q_BUS'], [13.81966011, -10.])
        npt.assert_almost_equal(state['V_BUS'], [1 + 0.j, 0.7236067 + 0.j])
        npt.assert_almost_equal(state['P_BR'], [13.81966011])

class TestSimulatorBus3OnlyLoad(unittest.TestCase):
    def setUp(self):
        case = {'baseMVA': 100}
        case['bus'] = np.array([
            [0, 3, 0, 0, 10, 1, 1],
            [1, 1, 0, 0, 10, 1.1, 0.9],
            [2, 1, 0, 0, 10, 1.1, 0.9]
        ])
        case['device'] = np.array([
            [0, 0, 0, 50, -50, 1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 1, 100, -100, 1, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, -1, 1, 100, -100, 1, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        case['branch'] = np.array([
            [0, 1,  1,   1, 0, 100, 0, 0, 1, -360, 360],
            [0, 2, .5, .25, 0, 100, 0, 0, 1, -360, 360]
        ])

        self.simulator = Simulator(case)

    def test_pfe(self):
        d = 5
        state, _, _, _ = self.simulator.transition([-10, -5], [], [], [], [])

        npt.assert_almost_equal(state['P_DEV'], [19.09042574, -10., -5.], decimal=d)
        npt.assert_almost_equal(state['Q_DEV'], [18.95504293, -10., -5.], decimal=d)
        npt.assert_almost_equal(state['P_BUS'], [19.09042574, -10., -5.], decimal=d)
        npt.assert_almost_equal(state['Q_BUS'], [18.95504293, -10., -5.], decimal=d)
        npt.assert_almost_equal(state['P_BR'],  [13.81966011, 5.27076563], decimal=d)
        npt.assert_almost_equal(state['V_BUS'], [1 * np.exp(0.j),
                                                 0.72360679 * np.exp(1.j * 4.4726e-18),
                                                 0.96088902 * np.exp(1.j * 0.0130)], decimal=d)


class TestSimulatorBus3LoadVRE(unittest.TestCase):
    def setUp(self):
        case = {'baseMVA': 100}
        case['bus'] = np.array([
            [0, 3, 0, 0, 10, 1, 1],
            [1, 1, 0, 0, 10, 1.1, 0.9],
            [2, 1, 0, 0, 10, 1.1, 0.9]
        ])
        case['device'] = np.array([
            [0, 0, 0, 50, -50, 1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 1, 100, -100, 1, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, -1, 1, 100, -100, 1, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        case['branch'] = np.array([
            [0, 1,  1,   1, 0, 100, 0, 0, 1, -360, 360],
            [0, 2, .5, .25, 0, 100, 0, 0, 1, -360, 360]
        ])

        self.simulator = Simulator(case)

    def test_pfe(self):
        d = 5
        state, _, _, _ = self.simulator.transition([], [], [], [], [])


