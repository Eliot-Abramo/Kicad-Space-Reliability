import warnings

import correlated_mc as cmc


class CorrelationGroupTests:
    def test_correlation_group_creation(self):
        group = cmc.CorrelationGroup(name="temp", references=["R1", "R2"], rho=0.8)
        assert group.name == "temp"
        assert group.references == ["R1", "R2"]
        assert group.rho == 0.8


class CorrelatedMCResultTests:
    def test_result_defaults(self):
        result = cmc.CorrelatedMCResult()
        assert len(result.samples) == 0
        assert result.mean == 0.0
        assert result.std == 0.0


class DeprecationWarningTests:
    def test_correlated_monte_carlo_returns_empty_result(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = cmc.correlated_monte_carlo([], [], mission_hours=100.0)
        assert isinstance(result, cmc.CorrelatedMCResult)
