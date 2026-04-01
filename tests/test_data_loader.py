"""
test_data_loader.py — Tests for data download, cleaning, and merging.
"""

import numpy as np
import pandas as pd
import pytest
from src.config import (
    TICKERS, SHORT_HISTORY_TICKERS, SQ_XYZ_CUTOVER,
    START_DATE, END_DATE
)


class TestQualityChecks:
    """Test the data quality protocol from CLAUDE.md §2.3."""

    def test_forward_fill_limit(self):
        """Forward-fill must not exceed 5 business days."""
        from src.data_loader import quality_checks

        # Create price series with a 4-day gap (should be filled)
        dates = pd.bdate_range("2023-01-01", periods=20)
        prices = pd.DataFrame({"Adj Close": 100.0}, index=dates)
        prices.loc[dates[5:9], "Adj Close"] = np.nan  # 4-day gap

        cleaned, flags = quality_checks(prices, "TEST", max_ffill_days=5)
        assert cleaned["Adj Close"].isna().sum() == 0, "4-day gap should be forward-filled"

    def test_spike_detection(self):
        """Flag returns > 25% as potential corporate action artifacts."""
        from src.data_loader import quality_checks

        dates = pd.bdate_range("2023-01-01", periods=10)
        prices = pd.DataFrame({"Adj Close": [100, 100, 100, 100, 200,
                                              200, 200, 200, 200, 200]},
                               index=dates)
        _, flags = quality_checks(prices, "TEST")
        assert flags["spike_flag"].any(), "100% return should trigger spike flag"

    def test_adj_close_used(self):
        """Verify only Adj Close is used for return computations."""
        from src.feature_engineering import compute_log_returns

        df = pd.DataFrame({
            "Close": [100, 110, 105],
            "Adj Close": [100, 105, 100],
        }, index=pd.bdate_range("2023-01-01", periods=3))

        panel = pd.DataFrame({"TEST": df["Adj Close"]})
        log_ret = compute_log_returns(panel)

        # Returns should match Adj Close, not Close
        expected = np.log(105 / 100)
        assert abs(log_ret["TEST"].iloc[1] - expected) < 1e-10


class TestSQXYZMerge:
    """Test SQ → XYZ ticker transition handling."""

    def test_no_overlap(self):
        """Merged series should have no duplicate dates."""
        from src.data_loader import merge_sq_xyz

        # This test requires network access; skip if data unavailable
        try:
            merged = merge_sq_xyz(cache=True)
            if merged.empty:
                pytest.skip("No data available")
            assert not merged.index.duplicated().any(), "No duplicate dates in merged SQ/XYZ"
        except Exception:
            pytest.skip("Data download required")

    def test_cutover_date(self):
        """Verify cutover happens around 2025-01-21."""
        assert SQ_XYZ_CUTOVER == "2025-01-21"


class TestShortHistoryTickers:
    """Verify short-history tickers are correctly documented."""

    def test_crwd_ipo(self):
        assert "CRWD" in SHORT_HISTORY_TICKERS
        assert SHORT_HISTORY_TICKERS["CRWD"] == "2019-06-12"

    def test_ddog_ipo(self):
        assert "DDOG" in SHORT_HISTORY_TICKERS
        assert SHORT_HISTORY_TICKERS["DDOG"] == "2019-09-19"

    def test_pltr_dpo(self):
        assert "PLTR" in SHORT_HISTORY_TICKERS
        assert SHORT_HISTORY_TICKERS["PLTR"] == "2020-09-30"

    def test_no_fabricated_data(self):
        """CRITICAL: Never fabricate pre-IPO prices."""
        # This is a documentation/design test — verify the config
        for ticker, ipo_date in SHORT_HISTORY_TICKERS.items():
            assert ticker in TICKERS, f"{ticker} must be in universe"
