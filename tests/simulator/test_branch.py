import numpy as np
import unittest

from gym_anm.simulator.components import TransmissionLine
from gym_anm.simulator.components.errors import BranchSpecError


class TestBranch(unittest.TestCase):
    def setUp(self):
        self.baseMVA = 10
        self.bus_ids = [1, 2]

    def test_simple_branch(self):
        for _ in range(100):
            spec = np.concatenate((np.array([1, 2]), np.random.uniform(0., 20, 5), np.random.uniform(0, 360, 1)))
            branch = TransmissionLine(spec, self.baseMVA, self.bus_ids)

            series = 1 / (spec[2] + 1.j * spec[3])
            shunt = 1.j * spec[4] / 2
            shift_rad = spec[7] * np.pi / 180
            tap = spec[6] * np.exp(1.j * shift_rad)

            self.assertEqual(branch.f_bus, spec[0])
            self.assertEqual(branch.t_bus, spec[1])
            self.assertEqual(branch.r, spec[2])
            self.assertEqual(branch.x, spec[3])
            self.assertEqual(branch.b, spec[4])
            self.assertEqual(branch.rate, spec[5] / self.baseMVA)
            self.assertEqual(branch.tap_magn, spec[6])
            self.assertEqual(branch.shift, shift_rad)
            self.assertEqual(branch.series, series)
            self.assertEqual(branch.shunt, shunt)
            self.assertEqual(branch.tap, tap)

    def test_bad_bus_ids(self):
        for f_bus in [None, 0, 3]:
            spec = np.array([f_bus, 2, 0.1, 0.2, 0.3, 10, 2, 20])
            with self.assertRaises(BranchSpecError):
                TransmissionLine(spec, self.baseMVA, self.bus_ids)

        for t_bus in [None, 0, 3]:
            spec = np.array([1, t_bus, 0.1, 0.2, 0.3, 10, 2, 20])
            with self.assertRaises(BranchSpecError):
                TransmissionLine(spec, self.baseMVA, self.bus_ids)

    def test_default_values(self):
        spec = np.array([1, 2, None, 1, None, None, None, None])
        br = TransmissionLine(spec, self.baseMVA, self.bus_ids)

        self.assertEqual(br.r, 0)
        self.assertEqual(br.b, 0)
        self.assertEqual(br.rate, np.inf)
        self.assertEqual(br.tap_magn, 1)
        self.assertEqual(br.shift, 0)

        spec = np.array([1, 2, 1, None, None, None, None, None])
        br = TransmissionLine(spec, self.baseMVA, self.bus_ids)
        self.assertEqual(br.x, 0)

    def test_infinite_impedance(self):
        spec = np.array([1, 2, 0, 0, None, None, None, None])
        with self.assertRaises(BranchSpecError):
            TransmissionLine(spec, self.baseMVA, self.bus_ids)

    def test_bad_rxb(self):
        for r in np.random.uniform(-10, -1, 10):
            spec = np.array([1, 2, r, 0.2, 0.3, 10, 2, 20])
            with self.assertRaises(BranchSpecError):
                TransmissionLine(spec, self.baseMVA, self.bus_ids)

        for x in np.random.uniform(-10, -1, 10):
            spec = np.array([1, 2, 0.1, x, 0.3, 10, 2, 20])
            with self.assertRaises(BranchSpecError):
                TransmissionLine(spec, self.baseMVA, self.bus_ids)

        for b in np.random.uniform(-10, -1, 10):
            spec = np.array([1, 2, 0.1, 0.2, b, 10, 2, 20])
            with self.assertRaises(BranchSpecError):
                TransmissionLine(spec, self.baseMVA, self.bus_ids)

    def test_bad_rate(self):
        for rate in np.random.uniform(-10, -1, 10):
            spec = np.array([1, 2, 0.1, 0.2, 0.3, rate, 2, 20])
            with self.assertRaises(BranchSpecError):
                TransmissionLine(spec, self.baseMVA, self.bus_ids)

    def test_bad_transformer(self):
        spec = np.array([1, 2, 0.1, 0.2, 0.3, 10, 0, 20])
        with self.assertRaises(BranchSpecError):
            TransmissionLine(spec, self.baseMVA, self.bus_ids)


        for tap_magn in np.random.uniform(-10, -1, 10):
            spec = np.array([1, 2, 0.1, 0.2, 0.3, 10, tap_magn, 0.])
            with self.assertRaises(BranchSpecError):
                TransmissionLine(spec, self.baseMVA, self.bus_ids)

        for shift in list(np.random.uniform(-360, -1, 50)) + list(np.random.uniform(360.1, 1000, 50)):
            spec = np.array([1, 2, 0.1, 0.2, 0.3, 10, 1., shift])
            with self.assertRaises(BranchSpecError):
                TransmissionLine(spec, self.baseMVA, self.bus_ids)

    def test_compute_currents(self):
        spec = np.array([1, 2, 0.1, 0.2, 0.3, 10, 2, 20])
        branch = TransmissionLine(spec, self.baseMVA, self.bus_ids)

        series = 1 / (spec[2] + 1.j * spec[3])
        shunt = 1.j * spec[4] / 2
        shift_rad = spec[7] * np.pi / 180
        tap = spec[6] * np.exp(1.j * shift_rad)

        for v_f, v_t in zip(np.random.uniform(-10, -10, 100), np.random.uniform(-10, -10, 100)):
            I_ij = (series + shunt) * v_f / np.abs(tap)**2 - series * v_t / np.conj(tap)
            I_ji = - series * v_f / tap + (series + shunt) * v_t

            branch.compute_currents(v_f, v_t)
            self.assertAlmostEqual(branch.i_from, I_ij, places=5)
            self.assertAlmostEqual(branch.i_to, I_ji, places=5)

    def test_compute_power_flows(self):
        spec = np.array([1, 2, 0.1, 0.2, 0.3, 10, 2, 20])
        branch = TransmissionLine(spec, self.baseMVA, self.bus_ids)

        for v_f, v_t in zip(np.random.uniform(-10, -10, 50), np.random.uniform(-10, -10, 50)):
            for i_f, i_t in zip(np.random.uniform(-10, -10, 50), np.random.uniform(-10, -10, 50)):
                branch.i_from = i_f
                branch.i_to = i_t
                s_f = v_f * np.conj(i_f)
                s_t = v_t * np.conj(i_t)
                branch.compute_power_flows(v_f, v_t)
                self.assertAlmostEqual(branch.p_from, s_f.real, places=5)
                self.assertAlmostEqual(branch.p_to, s_t.real, places=5)
                self.assertAlmostEqual(branch.q_from, s_f.imag,places=5)
                self.assertAlmostEqual(branch.q_to, s_t.imag, places=5)


if __name__ == '__main__':
    unittest.main()
