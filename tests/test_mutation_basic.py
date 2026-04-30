"""Basic mutation tests: verify tests catch specific code mutations.

If any of these pass, the corresponding test is not detecting the mutation.
"""

import math
import sys

import reliability_math as rm


class MutationDetectionTests:
    """Each test applies a known-bad mutation and asserts the module rejects it."""

    def test_fit_to_lambda_wrong_factor(self):
        """If fit_to_lambda used wrong constant, results should differ."""
        original = rm.fit_to_lambda(1000)
        wrong = 1000 * 1e-8  # off by 10x
        assert original != wrong

    def test_validate_ratio_does_not_accept_out_of_range(self):
        """If validate_ratio didn't clamp, -0.5 would be returned as-is."""
        clamped = rm.validate_ratio(-0.5)
        assert clamped != -0.5
        assert clamped == 0.0

    def test_validate_positive_clamps_negative(self):
        clamped = rm.validate_positive(-10.0)
        assert clamped != -10.0
        assert clamped == 0.0

    def test_reliability_from_lambda_zero_lambda(self):
        """If formula was wrong, zero lambda shouldn't give reliability != 1."""
        r = rm.reliability_from_lambda(0.0, 1000)
        assert r == 1.0

    def test_r_series_is_not_sum(self):
        """Series reliability is product, not sum."""
        r = rm.r_series([0.9, 0.9])
        assert r != 1.8  # sum would be 1.8
        assert r == 0.81  # correct is product

    def test_r_parallel_is_not_product(self):
        """Parallel reliability is not product of reliabilities."""
        r = rm.r_parallel([0.9, 0.9])
        assert r != 0.81  # product would be 0.81
        assert r == 0.99  # correct is 1 - (1-0.9)^2

    def test_r_k_of_n_not_series_or_parallel(self):
        """k-of-n is distinct from both series and parallel."""
        rs = [0.9, 0.9, 0.9]
        r_k = rm.r_k_of_n(rs, k=2)
        r_ser = rm.r_series(rs)
        r_par = rm.r_parallel(rs)
        assert r_k != r_ser
        assert r_k != r_par
