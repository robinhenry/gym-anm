import unittest
import numpy as np
import numpy.testing as npt

from gym_anm.envs import ANM6Easy
from gym_anm import MPCAgentConstant
from tests.base_test import BaseTest


class TestDCOPFAgent(BaseTest):

    def setUp(self):
        self.seed = 2020
        np.random.seed(self.seed)

        self.env = ANM6Easy()
        self.env.seed(self.seed)

        # Reset the environment, ensuring it is in a non-terminal state.
        done = True
        while done:
            self.env.reset()
            done = self.env.done

        self.safety_margin = 0.9
        self.B = self.env.simulator.Y_bus.imag.toarray()

        self.tolerance = 1e-5

    def test_ANM6Easy_horizon_1(self):
        """Test the DC OPF agent with a single timestep planning horizon."""
        agent = MPCAgentConstant(self.env.simulator, self.env.action_space,
                         self.safety_margin, planning_steps=1)

        for i in range(int(1e3)):
            a = agent.act(self.env)
            self._check_ANM6Easy_constraints(agent)

            _, _, done, _ = self.env.step(a)
            while done:
                self.env.reset()

    def test_ANM6Easy_horizon_3(self):
        """Test the DC OPF agent with a planning horizon of 3 timesteps."""
        agent = MPCAgentConstant(self.env.simulator, self.env.action_space,
                         self.safety_margin, planning_steps=3)

        for i in range(int(1e3)):
            a = agent.act(self.env)
            self._check_ANM6Easy_constraints(agent)

            _, _, done, _ = self.env.step(a)
            while done:
                self.env.reset()

    def test_ANM6Easy_horizon_20(self):
        """Test the DC OPF agent with a planning horizon of 20 timesteps."""
        agent = MPCAgentConstant(self.env.simulator, self.env.action_space,
                         self.safety_margin, planning_steps=20)

        for i in range(int(1e3)):
            a = agent.act(self.env)
            self._check_ANM6Easy_constraints(agent)

            _, _, done, _ = self.env.step(a)
            while done:
                self.env.reset()

    def _check_ANM6Easy_constraints(self, agent):

        # Extract OPF solution.
        V_ang = agent.V_bus_ang.value

        # C1) Bus DC power flow constraints.
        P = [self.B[0, 1] * (V_ang[0] - V_ang[1]),
             self.B[1, 0] * (V_ang[1] - V_ang[0]) + self.B[1, 2] * (V_ang[1] - V_ang[2]) + self.B[1, 3] * (V_ang[1] - V_ang[3]),
             self.B[2, 1] * (V_ang[2] - V_ang[1]) + self.B[2, 4] * (V_ang[2] - V_ang[4]) + self.B[2, 5] * (V_ang[2] - V_ang[5]),
             self.B[3, 1] * (V_ang[3] - V_ang[1]),
             self.B[4, 2] * (V_ang[4] - V_ang[2]),
             self.B[5, 2] * (V_ang[5] - V_ang[2])]
        P_bus = self._extract_expression_list(agent._create_p_bus_expressions(agent.P_dev))
        npt.assert_allclose(P, P_bus, atol=self.tolerance)

        # C2) New P_load = old P_load.
        for i in [1, 3, 5]:
            npt.assert_allclose(self.env.simulator.devices[i].p,
                                agent.P_dev.value[i],
                                atol=self.tolerance)

        # C3) P_gen and P_des are in [p_min, p_max].
        for i in [2, 4, 6]:
            self.assert_almost_less_than(self.env.simulator.devices[i].p_min,
                                         agent.P_dev.value[i])
            self.assert_almost_less_than(agent.P_dev.value[i],
                                         self.env.simulator.devices[i].p_max)

        # C4) P_gen <= P_max for each generator.
        for i in [2, 4]:
            self.assert_almost_less_than(agent.P_dev.value[i],
                                         self.env.simulator.devices[i].p_pot)

        # C5) New soc in [soc_min, soc_max].
        new_soc = self.env.simulator.devices[6].soc - agent.P_dev.value[6] \
                  * self.env.delta_t
        self.assert_almost_less_than(self.env.simulator.devices[6].soc_min,
                                     new_soc)
        self.assert_almost_less_than(new_soc,
                                     self.env.simulator.devices[6].soc_max)

        # C6) All voltage angles are in [- \pi, \pi].
        self.assert_almost_less_than([- np.pi] * 5, V_ang[1:])
        self.assert_almost_less_than(V_ang[1:], [np.pi] * 5)

        # C7) Angle V_0 = 0 (slack bus).
        self.assertAlmostEqual(0., V_ang[0])

    def _extract_expression_list(self, exprs):
        out = []
        for expr in exprs:
            if isinstance(expr, float):
                out.append(expr)
            else:
                out.append(expr.value)

        return out

    def assert_almost_less_than(self, a, b):
        atol = 1e-10
        if isinstance(a, (int, float)):
            self.assertTrue(np.allclose(a, b, atol=atol) or a < b)
        elif isinstance(a, (list, np.ndarray)):
            for i in range(len(a)):
                self.assertTrue(np.allclose(a[i], b[i], atol=atol) or a[i] < b[i])


if __name__ == '__main__':
    unittest.main()
