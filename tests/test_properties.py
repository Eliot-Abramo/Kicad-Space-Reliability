import reliability_math as rm
from hypothesis import given
from hypothesis import strategies as st

# Strategies that generate realistic engineering values
positive_floats = st.floats(min_value=1e-12, max_value=1e9, allow_nan=False, allow_infinity=False)
probabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
small_reliabilities = st.floats(min_value=1e-12, max_value=1.0, allow_nan=False, allow_infinity=False)
temperatures = st.floats(min_value=-250.0, max_value=200.0, allow_nan=False, allow_infinity=False)


class PropertyMathConversionsTests:
    @given(st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False))
    def test_fit_lambda_roundtrip(self, fit):
        lam = rm.fit_to_lambda(fit)
        fit_back = rm.lambda_to_fit(lam)
        assert abs(fit_back - fit) < 1e-6 or abs(fit_back - fit) / max(fit, 1) < 1e-9

    @given(positive_floats, positive_floats)
    def test_reliability_bounds(self, lam, hours):
        r = rm.reliability_from_lambda(lam, hours)
        assert 0.0 <= r <= 1.0

    @given(probabilities.filter(lambda r: 0 < r < 1), positive_floats)
    def test_lambda_reliability_roundtrip(self, r, hours):
        lam = rm.lambda_from_reliability(r, hours)
        r_back = rm.reliability_from_lambda(lam, hours)
        assert abs(r_back - r) < 1e-10

    @given(positive_floats)
    def test_mttf_is_reciprocal_of_lambda(self, lam):
        mttf = rm.mttf_from_lambda(lam)
        assert abs(mttf * lam - 1.0) < 1e-9


class PropertySystemTopologyTests:
    @given(st.lists(probabilities.filter(lambda r: 0 < r < 1), min_size=1, max_size=10))
    def test_r_series_is_at_most_min_component_reliability(self, rs):
        r_series = rm.r_series(rs)
        assert r_series <= min(rs) + 1e-12
        assert r_series >= 0

    @given(st.lists(probabilities.filter(lambda r: 0 < r < 1), min_size=1, max_size=10))
    def test_r_parallel_is_at_least_max_component_reliability(self, rs):
        r_parallel = rm.r_parallel(rs)
        assert r_parallel >= max(rs) - 1e-12
        assert r_parallel <= 1.0

    @given(st.lists(positive_floats, min_size=1, max_size=10))
    def test_lambda_series_equals_sum(self, lambdas):
        lam = rm.lambda_series(lambdas)
        assert abs(lam - sum(lambdas)) < 1e-15

    @given(st.lists(small_reliabilities, min_size=2, max_size=6))
    def test_k_of_n_between_series_and_parallel(self, rs):
        n = len(rs)
        r_series = rm.r_series(rs)
        r_parallel = rm.r_parallel(rs)
        for k in range(1, n + 1):
            r_k = rm.r_k_of_n(rs, k=k)
            assert r_k >= r_series - 1e-12
            assert r_k <= r_parallel + 1e-12

    @given(st.lists(probabilities.filter(lambda r: 0 < r < 1), min_size=2, max_size=6))
    def test_k_eq_1_is_full_parallel(self, rs):
        """k=1 means any 1 of N suffices = parallel."""
        r_k = rm.r_k_of_n(rs, k=1)
        r_par = rm.r_parallel(rs)
        assert abs(r_k - r_par) < 1e-12

    @given(st.lists(probabilities.filter(lambda r: 0 < r < 1), min_size=2, max_size=6))
    def test_k_eq_n_is_series(self, rs):
        """k=N means all N must work = series."""
        n = len(rs)
        r_k = rm.r_k_of_n(rs, k=n)
        r_ser = rm.r_series(rs)
        assert abs(r_k - r_ser) < 1e-12


class PropertyValidationTests:
    @given(temperatures)
    def test_validate_ratio_clamps(self, val):
        r = rm.validate_ratio(val)
        assert 0.0 <= r <= 1.0

    @given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    def test_validate_positive_clamps(self, val):
        r = rm.validate_positive(val)
        assert r >= 0.0

    @given(temperatures)
    def test_validate_temperature_above_absolute_zero(self, val):
        t = rm.validate_temperature(val)
        assert t >= -273.15
