import unittest

import component_swap as cs
import reliability_math


class SwapCandidateTests(unittest.TestCase):
    def test_swap_candidate_creation(self):
        cand = cs.SwapCandidate(
            name="SO-8 -> DIP-8",
            parameter="package",
            old_value="SO-8",
            new_value="DIP-8",
            lambda_before=1e-9,
            lambda_after=2e-9,
            delta_fit=1.0,
            delta_percent=100.0,
            improvement=False,
        )
        self.assertEqual(cand.name, "SO-8 -> DIP-8")
        self.assertEqual(cand.parameter, "package")
        self.assertFalse(cand.improvement)

    def test_swap_candidate_improvement_detected(self):
        cand = cs.SwapCandidate(
            name="DIP-8 -> SO-8",
            parameter="package",
            old_value="DIP-8",
            new_value="SO-8",
            lambda_before=2e-9,
            lambda_after=1e-9,
            delta_fit=-1.0,
            delta_percent=-50.0,
            improvement=True,
        )
        self.assertTrue(cand.improvement)
        self.assertLess(cand.delta_fit, 0)


class SwapAnalysisResultTests(unittest.TestCase):
    def test_result_creation(self):
        result = cs.SwapAnalysisResult(
            reference="U1",
            component_type="Integrated Circuit",
            current_fit=100.0,
            candidates=[],
            best_candidate=None,
            system_fit_before=0.0,
            system_fit_after_best=0.0,
        )
        self.assertEqual(result.reference, "U1")
        self.assertEqual(result.component_type, "Integrated Circuit")
        self.assertEqual(len(result.candidates), 0)

    def test_result_with_candidates(self):
        c1 = cs.SwapCandidate("A", "pkg", "old", "new", 1e-9, 2e-9, 1.0, 100.0, False)
        c2 = cs.SwapCandidate("B", "pkg", "old", "new", 2e-9, 1e-9, -1.0, -50.0, True)
        result = cs.SwapAnalysisResult(
            reference="U1",
            component_type="IC",
            current_fit=100.0,
            candidates=[c1, c2],
            best_candidate=c2,
            system_fit_before=100.0,
            system_fit_after_best=50.0,
        )
        self.assertEqual(len(result.candidates), 2)
        self.assertIs(result.best_candidate, c2)


class AnalyzePackageSwapsTests(unittest.TestCase):
    def test_analyze_package_swaps_returns_candidates(self):
        with unittest.mock.patch.object(
            reliability_math,
            "calculate_component_lambda",
            return_value={"lambda_total": 1e-9},
        ):
            result = cs.analyze_package_swaps(
                component_type="Integrated Circuit",
                base_params={"package": "QFP-48 (7x7mm)", "t_junction": 50.0},
            )
        self.assertIsInstance(result, cs.SwapAnalysisResult)
        self.assertEqual(result.component_type, "Integrated Circuit")


class AnalyzeTypeSwapsTests(unittest.TestCase):
    def test_analyze_type_swaps(self):
        result = cs.analyze_type_swaps(
            component_type="Resistor",
            base_params={"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0},
        )
        self.assertIsInstance(result, cs.SwapAnalysisResult)
        self.assertEqual(result.component_type, "Resistor")


class QuickSwapComparisonTests(unittest.TestCase):
    def test_quick_swap_comparison(self):
        with unittest.mock.patch.object(
            reliability_math,
            "calculate_component_lambda",
            return_value={"lambda_total": 2e-9},
        ):
            result = cs.quick_swap_comparison(
                component_type="Resistor",
                base_params={"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0},
                parameter="v_applied",
                new_value=2.5,
            )
        self.assertIn("parameter", result)
        self.assertIn("old_value", result)
        self.assertIn("new_value", result)
        self.assertIn("delta_fit", result)
        self.assertEqual(result["parameter"], "v_applied")


class RankAllSwapsTests(unittest.TestCase):
    def test_rank_all_swaps_no_components(self):
        result = cs.rank_all_swaps({})
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
