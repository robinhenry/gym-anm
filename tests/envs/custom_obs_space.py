import numpy as np
import unittest
import numpy.testing as npt

from gym_anm.envs import ANMEnv


class TestCustomObservationSpace(unittest.TestCase):

    def setUp(self):
        self.baseMVA = 10

        network = {
            'baseMVA': self.baseMVA,
            'bus': np.array([[0, 0, 50, 1.1, 0.9],
                             [1, 1, 50, 1.1, 0.9],
                             [2, 1, 100, 1., 1.]]),  # slack bus
            'branch': np.array([[0, 1, 0.1, 0.2, 0.3, 20, 1, 90],
                                [1, 2, 0.4, 0.5, 0.6, 20, 2, 0]]),
            'device': np.array([
                [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None],
                [1, 1, -1, 0.2, 0, -10, None, None, None, None, None, None, None, None, None],
                [2, 2, 2, None, 30, 0, 30, -30, None, None, None, None, None, None, None],
                [3, 2, 3, None, 50, -50, 50, -50, None, None, None, None, 100, 0, 0.9]])
        }
        self.network = network
        self.delta_t = 1
        self.lamb = 100
        self.gamma = 0.9

    def test_simple_builtin_obs_space(self):
        """Test an environment built by specifying a list of values for the obs space."""

        K = 0
        observation = [('bus_p', 'all', 'MW'), ('dev_q', [0, 2], 'pu'),
                       ('branch_s', 'all', 'pu')]
        env = ANMEnv(self.network, observation, K, self.delta_t, self.gamma,
                     self.lamb, None, None, None)

        env.init_state = lambda: np.random.rand(10)
        env.reset()

        # Check the environment has correctly stored the provided obs values.
        true_obs_values = [('bus_p', [0, 1, 2], 'MW'),
                           ('dev_q', [0, 2], 'pu'),
                           ('branch_s', [(0, 1), (1, 2)], 'pu')]
        self.assertEqual(env.obs_values, true_obs_values)

        # Check that the observation space bounds have been updated.
        true_upper_bounds = np.array([200, 0, 80, 20, 3, np.inf, np.inf])
        true_lower_bounds = np.array([-200, -10, -50, -20, -3, -np.inf, -np.inf])

        npt.assert_allclose(true_lower_bounds, env.observation_space.low)
        npt.assert_allclose(true_upper_bounds, env.observation_space.high)

        # Check that the correct observation vector is returned.
        ps = [100, -5, 60]
        env.simulator.buses[0].p = ps[0] / self.baseMVA
        env.simulator.buses[1].p = ps[1] / self.baseMVA
        env.simulator.buses[2].p = ps[2] / self.baseMVA

        qs = [-150, -20]
        env.simulator.devices[0].q = qs[0] / self.baseMVA
        env.simulator.devices[2].q = qs[1] / self.baseMVA

        branch_ss = [15, 25]
        env.simulator.branches[(0, 1)].s_apparent_max = branch_ss[0] / self.baseMVA
        env.simulator.branches[(1, 2)].s_apparent_max = branch_ss[1] / self.baseMVA

        env.simulator.state = env.simulator._gather_state()

        obs = env.observation(None)
        for i in range(3):
            self.assertEqual(obs[i], ps[i])
        for i in range(2):
            self.assertEqual(obs[3+i], qs[i] / self.baseMVA)
        for i in range(2):
            self.assertEqual(obs[5+i], branch_ss[i] / self.baseMVA)
