import numpy.testing as npt
import unittest


class BaseTest(unittest.TestCase):

    def assert_dict_all_close(self, actual, desired, places=5):
        super().assertIsInstance(actual, dict)
        super().assertIsInstance(desired, dict)

        # Check keys are equal.
        super().assertSetEqual(set(actual.keys()), set(desired.keys()))

        for k in actual.keys():
            a = actual[k]
            b = desired[k]
            if isinstance(a, dict) and isinstance(b, dict):
                self.assert_dict_all_close(a, b, places)
            else:
                npt.assert_allclose(a, b, rtol=10**(-places))
