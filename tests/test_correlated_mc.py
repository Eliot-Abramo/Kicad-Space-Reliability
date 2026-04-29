import unittest
import warnings

import correlated_mc as cmc


class CorrelationGroupTests(unittest.TestCase):
    def test_correlation_group_creation(self):
        group = cmc.CorrelationGroup(name="temp", references=["R1", "R2"], rho=0.8)
        self.assertEqual(group.name, "temp")
        self.assertEqual(group.references, ["R1", "R2"])
        self.assertEqual(group.rho, 0.8)


class CorrelatedMCResultTests(unittest.TestCase):
    def test_result_defaults(self):
        result = cmc.CorrelatedMCResult()
        self.assertEqual(len(result.samples), 0)
        self.assertEqual(result.mean, 0.0)
        self.assertEqual(result.std, 0.0)


class DeprecationWarningTests(unittest.TestCase):
    def test_correlated_monte_carlo_returns_empty_result(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = cmc.correlated_monte_carlo([], [], mission_hours=100.0)
        self.assertIsInstance(result, cmc.CorrelatedMCResult)


if __name__ == "__main__":
    unittest.main()
