import unittest
import numpy.testing as npt
import numpy as np

from gym_anm.simulator import Simulator
from gym_anm.simulator.solve_load_flow import solve_pfe_newton_raphson


class TestSimulatorTransition(unittest.TestCase):
    def setUp(self):
        self.delta_t = 0.5
        self.lamb = 100
        self.baseMVA = 1

        self.places = 4
        self.rtol = 1e-4

    def test_2bus_2dev(self):
        """Solve a single-branch 2-bus AC load flow."""
        # Network definition.
        network = {
            'baseMVA': self.baseMVA,
            'bus': np.array([[0, 0, 50, 1., 1.],
                             [1, 1, 50, 1.1, 0.9]]),
            'branch': np.array([[0, 1, 0.01, 0.1, 0., 32, 1, 0]]),
            'device': np.array([
                [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None,
                 None, None, None],  # slack
                [1, 1, -1, 0.2, 0, -10, None, None, None, None, None, None, None,
                 None, None]  # load
            ])
        }
        simulator = Simulator(network, self.delta_t, self.lamb)

        # Set device fixed power injections.
        N = 50
        for i, pq in enumerate(np.random.uniform(-1, 0, size=(N, 2))):
            simulator.devices[1].p = pq[0]
            simulator.devices[1].q = pq[1]

            # Set bus injections (same as device injections).
            simulator.buses[1].p = simulator.devices[1].p
            simulator.buses[1].q = simulator.devices[1].q

            # My own implementation.
            solve_pfe_newton_raphson(simulator)
            self._check_pfe_solution(simulator)

    def test_3bus_4dev(self):
        """Solve load flow with a loop network (no transformers)."""

        # Network definition.
        network = {
            'baseMVA': self.baseMVA,
            'bus': np.array([[0, 0, 50, 1., 1.],
                             [1, 1, 50, 1.1, 0.9],
                             [2, 1, 50, 1.1, 0.9]]),
            'branch': np.array([[0, 1, 0.01, 0.1, 0., 30, 1, 0],
                                [1, 2, 0.02, 0.3, 0.2, 30, 1, 0],
                                [2, 0, 0.05, 0.2, 0.1, 30, 1, 0]]),
            'device': np.array([
                [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None],  # slack
                [1, 1, -1, 0.2, 0, -10, None, None, None, None, None, None, None, None, None],  # load
                [2, 1, 1, None, 200, 0, 200, -200, None, None, None, None, None, None, None],  # gen
                [3, 2, 2, None, 200, 0, 200, -200, None, None, None, None, None, None, None],  # renewable
                [4, 2, 3, None, 200, -200, 200, -200, None, None, None, None, 100, 0, 0.9]  # storage
            ])
        }
        simulator = Simulator(network, self.delta_t, self.lamb)

        # Set device fixed power injections.
        N = 50
        pq_load = np.random.uniform(-1, 0, (N, 2))
        pq_gen = np.random.uniform(0, 1, (N, 2))
        pq_ren = np.random.uniform(0, 1, (N, 2))
        pq_des = np.random.uniform(-1, 1, (N, 2))
        for i in range(N):
            for j, pq in zip(range(1, 5),
                             [pq_load[i], pq_gen[i], pq_ren[i], pq_des[i]]):
                simulator.devices[j].p = pq[0]
                simulator.devices[j].q = pq[1]

            # Set bus power injections.
            simulator.buses[1].p = pq_load[i, 0] + pq_gen[i, 0]
            simulator.buses[1].q = pq_load[i, 1] + pq_gen[i, 1]
            simulator.buses[2].p = pq_ren[i, 0] + pq_des[i, 0]
            simulator.buses[2].q = pq_ren[i, 1] + pq_des[i, 1]

            # Solve power flow equations.
            solve_pfe_newton_raphson(simulator)

            # Check the solution.
            self._check_pfe_solution(simulator)

    def test_3bus_4dev_1transformer(self):
        """Solve load flow with a off-nominal transformer."""

        N_runs = 10
        for i in range(N_runs):
            tap = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(0, 50)

            # Network definition.
            network = {
                'baseMVA': self.baseMVA,
                'bus': np.array([[0, 0, 50, 1., 1.],
                                 [1, 1, 50, 1.1, 0.9],
                                 [2, 1, 50, 1.1, 0.9]]),
                'branch': np.array([[0, 1, 0.01, 0.1, 0., 30, 1, 0],
                                    [1, 2, 0.02, 0.3, 0.2, 30, tap, shift],
                                    [2, 0, 0.05, 0.2, 0.1, 30, 1, 0]]),
                'device': np.array([
                    [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None],  # slack
                    [1, 1, -1, 0.2, 0, -10, None, None, None, None, None, None, None, None, None],  # load
                    [2, 1, 1, None, 200, 0, 200, -200, None, None, None, None, None, None, None],  # gen
                    [3, 2, 2, None, 200, 0, 200, -200, None, None, None, None, None, None, None],  # renewable
                    [4, 2, 3, None, 200, -200, 200, -200, None, None, None, None, 100, 0, 0.9]  # storage
                ])
            }
            simulator = Simulator(network, self.delta_t, self.lamb)

            # Set device fixed power injections.
            N = 10
            pq_load = np.random.uniform(-1, 0, (N, 2))
            pq_gen = np.random.uniform(0, 1, (N, 2))
            pq_ren = np.random.uniform(0, 1, (N, 2))
            pq_des = np.random.uniform(-1, 1, (N, 2))
            for i in range(N):
                for j, pq in zip(range(1, 5),
                                 [pq_load[i], pq_gen[i], pq_ren[i], pq_des[i]]):
                    simulator.devices[j].p = pq[0]
                    simulator.devices[j].q = pq[1]

                # Set bus power injections.
                simulator.buses[1].p = pq_load[i, 0] + pq_gen[i, 0]
                simulator.buses[1].q = pq_load[i, 1] + pq_gen[i, 1]
                simulator.buses[2].p = pq_ren[i, 0] + pq_des[i, 0]
                simulator.buses[2].q = pq_ren[i, 1] + pq_des[i, 1]

                # Solve power flow equations.
                solve_pfe_newton_raphson(simulator)

                # Check the solution.
                self._check_pfe_solution(simulator)

    def test_reset(self):
        """Test reset() (and transition()) methods."""

        # Network definition.
        baseMVA = 10
        network = {
            'baseMVA': baseMVA,
            'bus': np.array([[0, 0, 50, 1., 1.],
                             [1, 1, 50, 1.1, 0.9],
                             [2, 1, 50, 1.1, 0.9]]),
            'branch': np.array([[0, 1, 0.01, 0.1, 0., 30, 1, 0],
                                [1, 2, 0.02, 0.3, 0.2, 30, 1, 0],
                                [2, 0, 0.05, 0.2, 0.1, 30, 1, 0]]),
            'device': np.array([
                [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None,
                 None, None, None],  # slack
                [1, 1, -1, 0.2, 0, -10, None, None, None, None, None, None, None,
                 None, None],  # load
                [2, 1, 1, None, 100, 0, 100, -100, None, None, None, None, None,
                 None, None],  # gen
                [3, 2, 2, None, 100, 0, 100, -100, None, None, None, None, None,
                 None, None],  # renewable
                [4, 2, 3, None, 100, -100, 100, -100, None, None, None, None,
                 100, 0, 0.9]  # storage
            ])
        }
        simulator = Simulator(network, self.delta_t, self.lamb)

        ps = [5, -5, 3, 6, -7]
        qs = [1, -1, 1, 2, -2]
        soc = [40]
        p_max = [4, 6]
        aux = [None, None]
        s0 = np.array(ps + qs + soc + p_max + aux)
        simulator.reset(s0)

        # Check results satisfy power flow equations.
        self._check_pfe_solution(simulator)

        # Check device power injections.
        for i in range(1, 5):
            self.assertAlmostEqual(simulator.devices[i].p, ps[i] / baseMVA)
            self.assertAlmostEqual(simulator.devices[i].q, qs[i] / baseMVA)

        # Check DES final SoC.
        self.assertAlmostEqual(simulator.devices[4].soc, soc[0] / baseMVA)

    def _check_pfe_solution(self, simulator):
        """
        Check that the current network state satisfies the AC power flow equations.

        This method is to be used after solving an AC load flow, in order to check
        the solution reached by the solver.
        """

        # Check slack bus has V = 1 + 0j.
        for bus in simulator.buses.values():
            if bus.is_slack:
                self.assertAlmostEqual(bus.v.real, 1., places=self.places)
                self.assertAlmostEqual(bus.v.imag, 0., places=self.places)

        # Check that each bus has P_i = \sum_d P_d and Q_i = \sum_d Q_d.
        p_bus = {i: 0 for i in simulator.buses.keys()}
        q_bus = {i: 0 for i in simulator.buses.keys()}
        for dev in simulator.devices.values():
            p_bus[dev.bus_id] += dev.p
            q_bus[dev.bus_id] += dev.q
        for bus in simulator.buses.values():
            self.assertAlmostEqual(p_bus[bus.id], bus.p)
            self.assertAlmostEqual(q_bus[bus.id], bus.q)

        # Check that each bus has S = P+jQ = VI^*.
        for bus in simulator.buses.values():
            s = bus.v * np.conj(bus.i)
            self.assertAlmostEqual(bus.p, s.real, places=self.places)
            self.assertAlmostEqual(bus.q, s.imag, places=self.places)

        # Check that I = YV (matrix notation).
        I_true = np.zeros(simulator.N_bus, dtype=np.complex)
        V = np.zeros(simulator.N_bus, dtype=np.complex)
        for bus in simulator.buses.values():
            I_true[bus.id] = bus.i
            V[bus.id] = bus.v
        I = np.dot(simulator.Y_bus.toarray(), V)
        npt.assert_allclose(I_true.real, I.real, rtol=self.rtol)
        npt.assert_allclose(I_true.imag, I.imag, rtol=self.rtol)

        # Check that branches have S_{ij} = V_i * I_{ij}^*.
        for branch in simulator.branches.values():
            # At the sending end.
            s_from = simulator.buses[branch.f_bus].v * np.conj(branch.i_from)
            self.assertAlmostEqual(s_from.real, branch.p_from,
                                   places=self.places)
            self.assertAlmostEqual(s_from.imag, branch.q_from,
                                   places=self.places)

            # At the receiving end.
            s_to = simulator.buses[branch.t_bus].v * np.conj(branch.i_to)
            self.assertAlmostEqual(s_to.real, branch.p_to, places=self.places)
            self.assertAlmostEqual(s_to.imag, branch.q_to, places=self.places)

        # Check that branches have I_{ij} = ... V_i + ... V_j.
        for br in simulator.branches.values():
            v_i = simulator.buses[br.f_bus].v
            v_t = simulator.buses[br.t_bus].v
            i_from_true = (br.series + br.shunt) / br.tap_magn ** 2 \
                          * v_i - br.series / np.conj(br.tap) * v_t
            i_to_true = - br.series / br.tap * v_i + (br.series + br.shunt) * v_t

            self.assertAlmostEqual(i_from_true.real, br.i_from.real,
                                   places=self.places)
            self.assertAlmostEqual(i_from_true.imag, br.i_from.imag,
                                   places=self.places)
            self.assertAlmostEqual(i_to_true.real, br.i_to.real,
                                   places=self.places)
            self.assertAlmostEqual(i_to_true.imag, br.i_to.imag,
                                   places=self.places)

        # Check the sign of |S_{ij}| for each branch.
        for br in simulator.branches.values():
            self.assertEqual(np.sign(br.s_apparent_max), np.sign(br.p_from))

        # Check that |S| = |S_{ij}| or |S| = |S_{ji}|.
        for br in simulator.branches.values():
            s_app = np.maximum(np.sqrt(br.p_from ** 2 + br.q_from ** 2),
                               np.sqrt(br.p_to ** 2 + br.q_to ** 2))
            if np.sign(br.s_apparent_max) >= 0:
                self.assertAlmostEqual(br.s_apparent_max, s_app,
                                       places=self.places)
            else:
                self.assertAlmostEqual(-br.s_apparent_max, s_app,
                                       places=self.places)

        return


if __name__ == '__main__':
    unittest.main()
