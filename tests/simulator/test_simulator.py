import unittest
import numpy.testing as npt
import numpy as np
import os
import copy

from gym_anm.simulator import Simulator


class TestSimulator(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))  # Set the working directory to the root.

        self.delta_t = 15

        network= {"baseMVA": 100.0}
        network["bus"] = np.array([
            [0, 3, 132, 1.04, 1.04],
            [1, 1, 33, 1.1, 0.9],
            [2, 1, 33, 1.1, 0.9],
        ])
        network["device"] = np.array([
            [0,  0, 0, 100, -90, 1, 80, -70, 0, 0, 0, 0, 0, 0,   0,   0],
            [1, -1, 0,   0,   0, 1,  0,  -5, 0, 0, 0, 0, 0, 0,   0,   0],
            [1,  3, 0,  10, -15, 1, 40,   0, 0, 0, 0, 0, 0, 0,   0,   0],
            [2,  2, 0,  20, -25, 1, 50,   0, 0, 0, 0, 0, 0, 0,   0,   0],
            [2,  4, 0,  30, -35, 1, 60, -40, 0, 0, 0, 0, 0, 0, 100, 0.9]
        ])
        network["branch"] = np.array([
            [0, 1, .1, .2, .3, 32, 1.1, 30, 1],
            [0, 2, .4, .5, .6, 24,   0,  0, 1],
            [1, 2, .7, .8, .9, 17,   0,  0, 1],
        ])
        self.simulator = Simulator(network, delta_t=self.delta_t)

    def test_Y_bus(self):
        y_bus = [
            [2.6285 - 4.1013j,  -3.3928 + 2.2401j, -0.9756 + 1.2195j],
            [0.2436 + 4.0583j, 2.6195 - 4.1080j, -0.6195 + 0.7080j],
            [-0.9756 + 1.2195j, -0.6195 + 0.7080j, 1.5951 - 1.1775j]
        ]
        npt.assert_almost_equal(self.simulator.Y_bus, y_bus, decimal=4)

    def test_load_case(self):
        npt.assert_almost_equal(self.simulator.baseMVA, 100.)
        self.assertEqual(self.simulator.N_bus, 3)
        self.assertEqual(self.simulator.N_branch, 3)
        self.assertEqual(self.simulator.N_device, 5)
        self.assertEqual(self.simulator.N_storage, 1)
        self.assertEqual(self.simulator.N_load, 1)
        self.assertEqual(self.simulator.N_gen, 3)

    def test_bus_bounds(self):
        npt.assert_almost_equal(self.simulator.buses[0].p_min, -70)
        npt.assert_almost_equal(self.simulator.buses[0].p_max, 80)
        npt.assert_almost_equal(self.simulator.buses[1].p_min, -5)
        npt.assert_almost_equal(self.simulator.buses[1].p_max, 40)
        npt.assert_almost_equal(self.simulator.buses[2].p_min, -40)
        npt.assert_almost_equal(self.simulator.buses[2].p_max, 110)
        npt.assert_almost_equal(self.simulator.buses[0].q_min, -90)
        npt.assert_almost_equal(self.simulator.buses[0].q_max, 100)
        npt.assert_almost_equal(self.simulator.buses[1].q_min, -15)
        npt.assert_almost_equal(self.simulator.buses[1].q_max, 10)
        npt.assert_almost_equal(self.simulator.buses[2].q_min, -60)
        npt.assert_almost_equal(self.simulator.buses[2].q_max, 50)

    def test_network_specs(self):
        specs = {'PMIN_BUS': [-70, -5, -40],
                 'PMAX_BUS': [80, 40, 110],
                 'QMIN_BUS': [-90, -15, -60],
                 'QMAX_BUS': [100, 10, 50],
                 'VMIN_BUS': [1.04, 0.9, 0.9],
                 'VMAX_BUS': [1.04, 1.1, 1.1],
                 'PMIN_DEV': [-70, -5, 0, 0, -40],
                 'PMAX_DEV': [80, 0, 40, 50, 60],
                 'QMIN_DEV': [-90, 0.0, -15, -25, -35],
                 'QMAX_DEV': [100, 0, 10, 20, 30],
                 'DEV_TYPE': [0, -1, 3, 2, 4],
                 'SMAX_BR': [32, 24, 17],
                 'SOC_MIN': [0],
                 'SOC_MAX': [100]}

        for name, spec in specs.items():
            npt.assert_almost_equal(self.simulator.specs[name], spec)

    def test_action_space(self):
        curt = np.array([[40, 0], [50, 0]])
        alpha = np.array([[60, -40]])
        q = np.array([[30, -35]])

        action_space = self.simulator.get_action_space()

        npt.assert_almost_equal(curt, action_space[0])
        npt.assert_almost_equal(alpha, action_space[1])
        npt.assert_almost_equal(q, action_space[2])

    def test_reward_no_penalty(self):
        p_dev = np.array([20, -50, 30, 5, 3])   # loss = 5 - 3 = 2
        p_pot = np.array([40, 10])              # loss = 15
        p_curt = np.array([30, 5])

        time_factor = self.delta_t / 60.

        simulator = copy.deepcopy(self.simulator)
        simulator.slack_dev.p = p_dev[0]
        simulator.loads[1].p = p_dev[1]
        simulator.gens[2].p = p_dev[2]
        simulator.gens[3].p = p_dev[3]
        simulator.storages[4].p = p_dev[4]

        # No penalty
        simulator.buses[0].v = 1.04
        for b in simulator.buses[1:]:
            b.v = 1.
        for b in simulator.branches:
            b.p_from = 0.1
            b.p_to = 0.1
            b.q_from = 0.1
            b.q_to = 0.1

        reward, e_loss, penalty = simulator._compute_reward(p_pot, p_curt)

        npt.assert_almost_equal(reward, - 17 * time_factor)
        npt.assert_almost_equal(e_loss, 17 * time_factor)
        npt.assert_almost_equal(penalty, 0)

    def test_reward_with_penalty(self):
        p_dev = np.array([20, -50, 30, 5, 3])  # loss = 5 - 3 = 2
        p_pot = np.array([40, 10])             # loss = 15
        p_curt = np.array([30, 5])

        time_factor = self.delta_t / 60.

        simulator = copy.deepcopy(self.simulator)
        simulator.slack_dev.p = p_dev[0]
        simulator.loads[1].p = p_dev[1]
        simulator.gens[2].p = p_dev[2]
        simulator.gens[3].p = p_dev[3]
        simulator.storages[4].p = p_dev[4]

        simulator.buses[0].v = 1.04
        for b in simulator.buses[1:]:
            b.v = 1.2

        simulator.branches[0].p_from = 35
        simulator.branches[0].q_from = 0
        simulator.branches[0].p_to = 0
        simulator.branches[0].q_to = 0

        simulator.branches[1].p_from = 0
        simulator.branches[1].q_from = 26
        simulator.branches[1].p_to = 0
        simulator.branches[1].q_to = 0

        simulator.branches[2].p_from = 0
        simulator.branches[2].q_from = 0
        simulator.branches[2].p_to = 18
        simulator.branches[2].q_to = 0

        reward, e_loss, penalty = simulator._compute_reward(p_pot, p_curt)

        npt.assert_almost_equal(reward, - 17 * time_factor - 260)
        npt.assert_almost_equal(e_loss, 17 * time_factor)
        npt.assert_almost_equal(penalty, 260)

if __name__ == '__main__':
    unittest.main()