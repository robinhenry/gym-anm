import unittest
import numpy as np
import numpy.testing as npt

from gym_anm.simulator import Simulator
from tests.base_test import BaseTest


class TestSimulator(BaseTest):
    def setUp(self):
        self.baseMVA = 10

        network = {
            'baseMVA': self.baseMVA,
            'bus': np.array([[0, 1, 50, 1.1, 0.9],
                             [2, 1, 50, 1.1, 0.9],
                             [1, 0, 100, 1., 1.]]),  # slack bus
            'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90],
                                [1, 2, 0.4, 0.5, 0.6, 20, 2, 0]]),
            'device': np.array([
            [1, 0, -1, 0.2, 0, -10, None, None, None, None, None, None, None, None, None],
            [0, 1, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None],
            [2, 2, 2, None, 30, 0, 30, -30, None, None, None, None, None, None, None],
            [3, 2, 3, None, 50, -50, 50, -50, None, None, None, None, 100, 0, 0.9]])
        }
        self.network = network
        self.delta_t = 1
        self.lamb = 100

        self.simulator = Simulator(self.network, self.delta_t, self.lamb)

        self.places = 5
        self.rtol = 1e-5

    def test_load_case(self):
        # Check lists of bus, branch, and device indices.
        self.assertListEqual([0, 1, 2], list(self.simulator.buses.keys()))
        self.assertListEqual([(0, 1), (1, 2)], list(self.simulator.branches.keys()))
        self.assertListEqual([0, 1, 2, 3], list(self.simulator.devices.keys()))

        # Check number of elements in each set.
        self.assertEqual(3, self.simulator.N_bus)
        self.assertEqual(4, self.simulator.N_device)
        self.assertEqual(1, self.simulator.N_load)
        self.assertEqual(1, self.simulator.N_non_slack_gen)
        self.assertEqual(1, self.simulator.N_des)

    def test_admittance_matrix(self):
        # Series and shunt admittances, transformer tap ratios.
        y01 = 1. / (0.1 + 1.j * 0.2)
        y01_sh = 1.j * 0.3 / 2.
        t01 = np.exp(1.j * np.pi / 2.)
        y12 = 1. / (0.4 + 1.j * 0.5)
        y12_sh = 1.j * 0.6 / 2.
        t12 = 2

        Y00 = (y01 + y01_sh) / (np.abs(t01) ** 2)
        Y01 = - y01 / np.conj(t01)
        Y10 = - y01 / t01
        Y11 = (y12 + y12_sh) / 4 + (y01 + y01_sh)
        Y12 = - y12 / np.conj(t12)
        Y21 = - y12 / t12
        Y22 = y12 + y12_sh

        Y = np.array([[Y00, Y01, 0],
                      [Y10, Y11, Y12],
                      [0, Y21, Y22]])

        npt.assert_allclose(Y, self.simulator.Y_bus.toarray(), rtol=self.rtol)

    def test_bus_bounds(self):
        true_bus_p_min = np.array([-10, -200, -50]) / self.baseMVA
        true_bus_p_max = np.array([0, 200, 80]) / self.baseMVA
        true_bus_q_min = np.array([-2, -200, -80]) / self.baseMVA
        true_bus_q_max = np.array([0, 200, 80]) / self.baseMVA

        for i, bus in enumerate(self.simulator.buses.values()):
            self.assertEqual(bus.p_min, true_bus_p_min[i])
            self.assertEqual(bus.p_max, true_bus_p_max[i])
            self.assertEqual(bus.q_min, true_bus_q_min[i])
            self.assertEqual(bus.q_max, true_bus_q_max[i])

    def test_get_action_space(self):
        P_gen_true = {2: np.array([0, 30])}
        Q_gen_true = {2: np.array([-30, 30])}
        P_des_true = {3: np.array([-50, 50])}
        Q_des_true = {3: np.array([-50, 50])}

        P_gen, Q_gen, P_des, Q_des = self.simulator.get_action_space()
        npt.assert_equal(P_gen_true, P_gen)
        npt.assert_equal(Q_gen_true, Q_gen)
        npt.assert_equal(P_des_true, P_des)
        npt.assert_equal(Q_des_true, Q_des)

    def test_get_state_space_buses(self):
        bus_p = {0: {'MW': (-10, 0), 'pu': (-10/self.baseMVA, 0)},
                 1: {'MW': (-200, 200), 'pu': (-200/self.baseMVA, 200/self.baseMVA)},
                 2: {'MW': (-50, 80), 'pu': (-50/self.baseMVA, 80/self.baseMVA)}}
        bus_q = {0: {'MVAr': (-2, 0), 'pu': (-2/self.baseMVA, 0)},
                 1: {'MVAr': (-200, 200), 'pu': (-200/self.baseMVA, 200/self.baseMVA)},
                 2: {'MVAr': (-80, 80), 'pu': (-80/self.baseMVA, 80/self.baseMVA)}}
        bus_v_magn = {0: {'pu': (-np.inf, np.inf), 'kV': (-np.inf, np.inf)},
                      1: {'pu': (1., 1.), 'kV': (100, 100)},
                      2: {'pu': (-np.inf, np.inf), 'kV': (-np.inf, np.inf)}}
        bus_v_ang = {0: {'degree': (-180, 180), 'rad': (-np.pi, np.pi)},
                     1: {'degree': (0., 0.), 'rad': (0., 0.)},
                     2: {'degree': (-180, 180), 'rad': (-np.pi, np.pi)}}
        bus_i_magn = {0: {'pu': (-np.inf, np.inf), 'kA': (-np.inf, np.inf)},
                      1: {'pu': (-np.inf, np.inf), 'kA': (-np.inf, np.inf)},
                      2: {'pu': (-np.inf, np.inf), 'kA': (-np.inf, np.inf)}}
        bus_i_ang = {0: {'degree': (-180, 180), 'rad': (-np.pi, np.pi)},
                     1: {'degree': (-180, 180), 'rad': (-np.pi, np.pi)},
                     2: {'degree': (-180, 180), 'rad': (-np.pi, np.pi)}}

        self.assert_dict_all_close(self.simulator.state_bounds['bus_p'], bus_p)
        self.assert_dict_all_close(self.simulator.state_bounds['bus_q'], bus_q)
        self.assert_dict_all_close(self.simulator.state_bounds['bus_v_magn'], bus_v_magn)
        self.assert_dict_all_close(self.simulator.state_bounds['bus_v_ang'], bus_v_ang)
        self.assert_dict_all_close(self.simulator.state_bounds['bus_i_magn'], bus_i_magn)
        self.assert_dict_all_close(self.simulator.state_bounds['bus_i_ang'], bus_i_ang)

    def test_get_state_space_devices(self):
        dev_p = {0: {'MW': (-200, 200), 'pu': (-200/self.baseMVA, 200/self.baseMVA)},
                 1: {'MW': (-10, 0), 'pu': (-10/self.baseMVA, 0)},
                 2: {'MW': (0, 30), 'pu': (0, 30/self.baseMVA)},
                 3: {'MW': (-50, 50), 'pu': (-50/self.baseMVA, 50/self.baseMVA)}}
        dev_q = {0: {'MVAr': (-200, 200), 'pu': (-200/self.baseMVA, 200/self.baseMVA)},
                 1: {'MVAr': (-2, 0), 'pu': (-2/self.baseMVA, 0)},
                 2: {'MVAr': (-30, 30), 'pu': (-30/self.baseMVA, 30/self.baseMVA)},
                 3: {'MVAr': (-50, 50), 'pu': (-50/self.baseMVA, 50/self.baseMVA)}}
        des_soc = {3: {'MWh': (0, 100), 'pu': (0, 100/self.baseMVA)}}
        gen_p_max = {2: {'MW': (0, 30), 'pu': (0, 30/self.baseMVA)}}

        self.assert_dict_all_close(self.simulator.state_bounds['dev_p'], dev_p)
        self.assert_dict_all_close(self.simulator.state_bounds['dev_q'], dev_q)
        self.assert_dict_all_close(self.simulator.state_bounds['des_soc'], des_soc)
        self.assert_dict_all_close(self.simulator.state_bounds['gen_p_max'], gen_p_max)

    def test_get_state_space_branches(self):
        branch_p = {(0, 1): {'pu': (-np.inf, np.inf), 'MW': (-np.inf, np.inf)},
                    (1, 2): {'pu': (-np.inf, np.inf), 'MW': (-np.inf, np.inf)}}
        branch_q = {(0, 1): {'pu': (-np.inf, np.inf), 'MVAr': (-np.inf, np.inf)},
                    (1, 2): {'pu': (-np.inf, np.inf), 'MVAr': (-np.inf, np.inf)}}
        branch_s = {(0, 1): {'pu': (-np.inf, np.inf), 'MVA': (-np.inf, np.inf)},
                    (1, 2): {'pu': (-np.inf, np.inf), 'MVA': (-np.inf, np.inf)}}
        branch_i_magn = {(0, 1): {'pu': (-np.inf, np.inf), 'kA': (-np.inf, np.inf)},
                         (1, 2): {'pu': (-np.inf, np.inf), 'kA': (-np.inf, np.inf)}}

        self.assert_dict_all_close(self.simulator.state_bounds['branch_p'], branch_p)
        self.assert_dict_all_close(self.simulator.state_bounds['branch_q'], branch_q)
        self.assert_dict_all_close(self.simulator.state_bounds['branch_s'], branch_s)
        self.assert_dict_all_close(self.simulator.state_bounds['branch_i_magn'], branch_i_magn)

    def test_get_bus_total_injections(self):
        n = 100
        dev0_p = np.random.uniform(-100, 100, n) / self.baseMVA
        dev0_q = np.random.uniform(-100, 100, n) / self.baseMVA
        dev1_p = np.random.uniform(-10, 0, n) / self.baseMVA
        dev1_q = np.random.uniform(-2, 0, n) / self.baseMVA
        dev2_p = np.random.uniform(0, 30, n) / self.baseMVA
        dev2_q = np.random.uniform(-30, 30, n) / self.baseMVA
        dev3_p = np.random.uniform(-50, 50, n) / self.baseMVA
        dev3_q = np.random.uniform(-50, 50, n) / self.baseMVA

        for i in range(n):
            self.simulator.devices[0].p = dev0_p[i]
            self.simulator.devices[0].q = dev0_q[i]
            self.simulator.devices[1].p = dev1_p[i]
            self.simulator.devices[1].q = dev1_q[i]
            self.simulator.devices[2].p = dev2_p[i]
            self.simulator.devices[2].q = dev2_q[i]
            self.simulator.devices[3].p = dev3_p[i]
            self.simulator.devices[3].q = dev3_q[i]

            self.simulator._get_bus_total_injections()

            self.assertAlmostEqual(self.simulator.buses[0].p, dev1_p[i])
            self.assertAlmostEqual(self.simulator.buses[1].p, dev0_p[i])
            self.assertAlmostEqual(self.simulator.buses[2].p, dev2_p[i] + dev3_p[i])
            self.assertAlmostEqual(self.simulator.buses[0].q, dev1_q[i])
            self.assertAlmostEqual(self.simulator.buses[1].q, dev0_q[i])
            self.assertAlmostEqual(self.simulator.buses[2].q, dev2_q[i] + dev3_q[i])

    def test_get_rendering_specs(self):
        bus_p = {0: {'MW': (-10, 0), 'pu': (-10/self.baseMVA, 0)},
                 1: {'MW': (-200, 200), 'pu': (-200/self.baseMVA, 200/self.baseMVA)},
                 2: {'MW': (-50, 80), 'pu': (-50/self.baseMVA, 80/self.baseMVA)}}
        bus_q = {0: {'MVAr': (-2, 0), 'pu': (-2/self.baseMVA, 0)},
                 1: {'MVAr': (-200, 200), 'pu': (-200/self.baseMVA, 200/self.baseMVA)},
                 2: {'MVAr': (-80, 80), 'pu': (-80/self.baseMVA, 80/self.baseMVA)}}
        dev_p = {0: {'MW': (-200, 200),
                     'pu': (-200 / self.baseMVA, 200 / self.baseMVA)},
                 1: {'MW': (-10, 0), 'pu': (-10 / self.baseMVA, 0)},
                 2: {'MW': (0, 30), 'pu': (0, 30 / self.baseMVA)},
                 3: {'MW': (-50, 50),
                     'pu': (-50 / self.baseMVA, 50 / self.baseMVA)}}
        dev_q = {0: {'MVAr': (-200, 200),
                     'pu': (-200 / self.baseMVA, 200 / self.baseMVA)},
                 1: {'MVAr': (-2, 0), 'pu': (-2 / self.baseMVA, 0)},
                 2: {'MVAr': (-30, 30),
                     'pu': (-30 / self.baseMVA, 30 / self.baseMVA)},
                 3: {'MVAr': (-50, 50),
                     'pu': (-50 / self.baseMVA, 50 / self.baseMVA)}}
        bus_v = {0: {'pu': (0.9, 1.1), 'kV': (45., 55.)},
                 1: {'pu': (1., 1.), 'kV': (100., 100.)},
                 2: {'pu': (0.9, 1.1), 'kV': (45., 55.)}}
        des_soc = {3: {'pu': (0., 10), 'MWh': (0., 100)}}
        branch_s = {(0, 1): {'pu': (0, 2), 'MVA': (0, 20)},
                    (1, 2): {'pu': (0, 2), 'MVA': (0, 20)}}

        specs = self.simulator.get_rendering_specs()

        self.assert_dict_all_close(specs['bus_p'], bus_p)
        self.assert_dict_all_close(specs['bus_q'], bus_q)
        self.assert_dict_all_close(specs['dev_p'], dev_p)
        self.assert_dict_all_close(specs['dev_q'], dev_q)
        self.assert_dict_all_close(specs['bus_v'], bus_v)
        self.assert_dict_all_close(specs['des_soc'], des_soc)
        self.assert_dict_all_close(specs['branch_s'], branch_s)

    def test_reward_no_penalty(self):
        p_dev = np.array([20, -5, 20, -30]) / self.baseMVA  # cost = 35
        p_max = 25 / self.baseMVA   # cost = 5
        v_bus = np.array([1., 1., 1.])
        s_branch = np.array([10, 10]) / self.baseMVA
        delta_t = 0.5
        true_r = - 40 * delta_t / self.baseMVA

        self.simulator.delta_t = delta_t
        self.simulator.devices[2].p_pot = p_max
        for i in range(4):
            self.simulator.devices[i].p = p_dev[i]
        for i, idx in enumerate([(0, 1), (1, 2)]):
            self.simulator.branches[idx].s_apparent_max = s_branch[i]
        for i in range(3):
            self.simulator.buses[i].v = v_bus[i]

        reward, e_loss, penalty = self.simulator._compute_reward()

        self.assertAlmostEqual(reward, true_r, places=self.places)
        self.assertAlmostEqual(e_loss, - true_r, places=self.places)
        self.assertEqual(penalty, 0)

    def test_reward_with_penalty(self):
        p_dev = np.array([20, -5, 20, -30]) / self.baseMVA  # cost = (35) / baseMVA
        p_max = 25 / self.baseMVA   # cost = 5 / baseMVA
        v_bus = np.array([1.2, 1., 0.8])  # cost = 0.2
        s_branch = np.array([30, 40]) / self.baseMVA  # cost = 30 / baseMVA
        delta_t = 0.5
        true_e_loss = 40 * delta_t / self.baseMVA
        true_penalty = self.lamb * delta_t * (0.2 + 30 / self.baseMVA)
        true_r = - (true_e_loss + true_penalty)

        self.simulator.delta_t = delta_t
        self.simulator.devices[2].p_pot = p_max
        for i in range(4):
            self.simulator.devices[i].p = p_dev[i]
        for i, idx in enumerate([(0, 1), (1, 2)]):
            self.simulator.branches[idx].s_apparent_max = s_branch[i]
        for i in range(3):
            self.simulator.buses[i].v = v_bus[i]

        reward, e_loss, penalty = self.simulator._compute_reward()

        self.assertAlmostEqual(e_loss, true_e_loss, places=self.places)
        self.assertAlmostEqual(penalty, true_penalty, places=self.places)
        self.assertAlmostEqual(reward, true_r, places=self.places)


if __name__ == '__main__':
    unittest.main()
