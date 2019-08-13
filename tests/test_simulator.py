import unittest
import numpy.testing as npt


class TestSimulator(unittest.TestCase):
    def setUp(self):
        pass

    def test_check_casefile(self):
        raise NotImplementedError

    def test_init(self):
        raise NotImplementedError

    def test_load_case(self):
        raise NotImplementedError

    def test_admittance_matrix(self):
        raise NotImplementedError

    def test_compute_bus_bounds(self):
        raise NotImplementedError

    def test_reset(self):
        raise NotImplementedError

    def test_get_action_space(self):
        raise NotImplementedError

    def test_transition(self):
        raise NotImplementedError

    def test_manage_storage(self):
        raise NotImplementedError

    def test_get_bus_total_injections(self):
        raise NotImplementedError

    def test_solve_pfe(self):
        raise NotImplementedError

    def test_get_branch_current(self):
        raise NotImplementedError

    def test_current_from_voltage(self):
        raise NotImplementedError

    def test_compute_branch_pq(self):
        raise NotImplementedError

    def test_get_reward(self):
        raise NotImplementedError

    def test_get_penalty(self):
        raise NotImplementedError

