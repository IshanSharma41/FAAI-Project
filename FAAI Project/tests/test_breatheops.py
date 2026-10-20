"""
Deterministic test suite for BreatheOps AI Phase 1.

Covers:
1. Loading built-in demo data.
2. Loading a valid CSV.
3. Missing required columns.
4. Missing pollutant values (safe handling without arbitrary imputations).
5. Invalid timestamps.
6. A dataset containing a detectable injected spike.
7. A dataset where no anomaly is detected.
8. A one-row dataset.
"""

from io import StringIO
import numpy as np
import pandas as pd
import pytest

from breatheops.analysis import (
    calculate_robust_baseline,
    detect_anomalies_mad,
    AnomalyDetectionResult,
)
from breatheops.data_loader import (
    load_air_quality_csv,
    normalize_column_name,
    DataValidationReport,
)
from breatheops.demo_data import generate_demo_data


class TestDemoDataLoader:
    """Tests for synthetic demo data generation."""

    def test_load_builtin_demo_data(self):
        """Test generating built-in demo data with fixed seed."""
        df = generate_demo_data(seed=42, days=7, inject_spike=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7 * 24 * 4  # 4 stations, 7 days, 24 hours = 672 rows
        assert "timestamp" in df.columns
        assert "station" in df.columns
        assert "PM2.5" in df.columns
        assert "PM10" in df.columns
        assert "TEMP" in df.columns
        assert "WSPM" in df.columns

        # Verify multiple stations exist
        stations = sorted(df["station"].unique())
        assert len(stations) == 4
        assert "Aotizhongxin" in stations

        # Verify synthetic flag
        assert df.attrs.get("is_synthetic") is True

        # Test CSV export and re-loading
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        loaded_df, report = load_air_quality_csv(csv_buffer, is_synthetic=True)
        assert report.is_valid is True
        assert report.total_rows == len(df)
        assert report.usable_rows == len(df)
        assert report.errors == []
        assert len(report.available_pollutants) == 6


class TestDataValidation:
    """Tests for CSV loading, column normalization, and error handling."""

    def test_load_valid_csv(self):
        """Test loading a clean valid CSV with date components and varying column cases."""
        csv_data = """Year, Month, Day, Hour, Station, pm2.5, PM10, TEMP, wspm
2023, 1, 1, 0, Station_A, 35.2, 55.1, -1.5, 2.1
2023, 1, 1, 1, Station_A, 38.0, 60.4, -2.0, 1.8
2023, 1, 1, 2, Station_A, 32.5, 50.0, -2.2, 2.4
"""
        df, report = load_air_quality_csv(StringIO(csv_data))

        assert report.is_valid is True
        assert report.errors == []
        assert report.total_rows == 3
        assert report.usable_rows == 3
        assert "PM2.5" in df.columns
        assert "PM10" in df.columns
        assert "TEMP" in df.columns
        assert "WSPM" in df.columns
        assert "timestamp" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_missing_required_columns_temporal(self):
        """Test handling CSV missing both timestamp and date component columns."""
        csv_data = """station, PM2.5, PM10, TEMP
Station_A, 45.0, 70.0, 12.0
Station_A, 50.0, 80.0, 13.5
"""
        df, report = load_air_quality_csv(StringIO(csv_data))

        assert df is None
        assert report.is_valid is False
        assert len(report.errors) > 0
        assert any("temporal" in err.lower() or "missing" in err.lower() for err in report.errors)
        assert "year" in report.missing_required_columns or "month" in report.missing_required_columns

    def test_missing_required_columns_pollutants(self):
        """Test handling CSV containing timestamps and weather but no pollutant columns."""
        csv_data = """timestamp, station, TEMP, PRES, RAIN
2023-01-01 00:00:00, Station_A, 5.0, 1013.2, 0.0
2023-01-01 01:00:00, Station_A, 4.8, 1013.0, 0.0
"""
        df, report = load_air_quality_csv(StringIO(csv_data))

        assert df is None
        assert report.is_valid is False
        assert any("pollutant" in err.lower() for err in report.errors)

    def test_missing_pollutant_values(self):
        """Test that missing pollutant values are handled safely without arbitrary imputation."""
        csv_data = """timestamp, station, PM2.5, PM10
2023-01-01 00:00:00, Station_A, 45.0, 60.0
2023-01-01 01:00:00, Station_A, , 62.0
2023-01-01 02:00:00, Station_A, NaN, 58.0
2023-01-01 03:00:00, Station_A, 48.0, NA
"""
        df, report = load_air_quality_csv(StringIO(csv_data))

        assert report.is_valid is True
        assert report.missing_pollutant_counts["PM2.5"] == 2
        assert report.missing_pollutant_percentages["PM2.5"] == 50.0
        assert report.missing_pollutant_counts["PM10"] == 1

        # Verify values remain NaN and were not arbitrarily filled with zeros or mean
        assert pd.isna(df.loc[1, "PM2.5"])
        assert pd.isna(df.loc[2, "PM2.5"])
        assert df.loc[0, "PM2.5"] == 45.0
        assert df.loc[3, "PM2.5"] == 48.0

    def test_invalid_timestamps(self):
        """Test handling rows with unparseable or corrupted timestamps."""
        csv_data = """timestamp, station, PM2.5
2023-01-01 00:00:00, Station_A, 40.0
invalid-date-string, Station_A, 42.0
2023-01-01 02:00:00, Station_A, 44.0
2023-13-45 99:99:99, Station_A, 46.0
"""
        df, report = load_air_quality_csv(StringIO(csv_data))

        assert report.is_valid is True
        assert report.total_rows == 4
        assert report.invalid_timestamp_count == 2
        assert report.usable_rows == 2
        assert len(report.warnings) > 0
        assert any("invalid or unparseable timestamps" in w for w in report.warnings)
        assert len(df) == 2

    def test_one_row_dataset(self):
        """Test handling single-row dataset and verifying limitation warnings."""
        csv_data = """timestamp, station, PM2.5, PM10, TEMP
2023-01-01 00:00:00, Station_A, 55.4, 82.1, 10.2
"""
        df, report = load_air_quality_csv(StringIO(csv_data))

        assert report.is_valid is True
        assert report.is_single_row is True
        assert report.total_rows == 1
        assert any("single row" in w.lower() or "only 1 observation" in w.lower() for w in report.warnings)

        # Pass to anomaly detection
        result = detect_anomalies_mad(df, pollutant="PM2.5", station="Station_A")
        assert result.sufficient_data is False
        assert result.anomaly_count == 0
        assert any("single" in w.lower() or "insufficient" in w.lower() for w in result.warnings)
        assert "insufficient historical evidence" in result.plain_language_finding.lower()


class TestAnomalyDetection:
    """Tests for statistical anomaly detection logic."""

    def test_detectable_injected_spike(self):
        """Verify that the deliberately injected spike in synthetic demo data is reliably detected."""
        # Seed 42 has a spike on Aotizhongxin between hours 160 and 170
        demo_df = generate_demo_data(seed=42, days=14, inject_spike=True, spike_station="Aotizhongxin")

        result = detect_anomalies_mad(
            df=demo_df,
            pollutant="PM2.5",
            station="Aotizhongxin",
            k_multiplier=3.0,
            min_observations=24,
        )

        assert result.sufficient_data is True
        assert result.anomaly_count > 0
        assert result.threshold_value is not None
        assert result.baseline_median is not None
        assert result.anomalies_df is not None
        assert len(result.anomalies_df) == result.anomaly_count

        # The peak in the injected spike reached > 250 µg/m³
        max_detected = result.anomalies_df["PM2.5"].max()
        assert max_detected > 200.0

        # Verify that all detected anomalies exceed the threshold
        assert (result.anomalies_df["PM2.5"] > result.threshold_value).all()

        # Check plain language finding
        assert "unusual pollution measurement(s) detected" in result.plain_language_finding
        assert result.non_regulatory_disclaimer in result.summary_markdown()

    def test_no_anomaly_detected(self):
        """Test dataset with stable, non-spiking values where no anomaly is flagged."""
        # Create deterministic synthetic normal readings with small variance
        rng = np.random.default_rng(123)
        hours = 72  # 3 days
        timestamps = pd.date_range("2023-01-01 00:00:00", periods=hours, freq="h")
        clean_pm25 = np.clip(35.0 + rng.normal(0, 2.5, hours), 25.0, 45.0)

        clean_df = pd.DataFrame({
            "timestamp": timestamps,
            "station": "Clean_Station",
            "PM2.5": clean_pm25,
        })

        result = detect_anomalies_mad(
            df=clean_df,
            pollutant="PM2.5",
            station="Clean_Station",
            k_multiplier=3.5,  # 3.5 MAD threshold
            min_observations=24,
        )

        assert result.sufficient_data is True
        assert result.anomaly_count == 0
        assert len(result.anomalies_df) == 0

        # Verify explanatory finding without claiming the air is guaranteed safe
        assert "no statistical anomalies were detected" in result.plain_language_finding.lower()
        assert "not a declaration that the air is clean or safe" in result.plain_language_finding.lower()

    def test_zero_mad_fallback(self):
        """Test that series with zero MAD falls back gracefully without division-by-zero."""
        # Series where more than 50% values are identical (MAD=0)
        values = np.array([10.0] * 50 + [12.0, 15.0, 25.0, 30.0])
        median, raw_mad, dispersion_name, scale = calculate_robust_baseline(values)

        assert median == 10.0
        assert raw_mad == 0.0
        assert "Fallback" in dispersion_name
        assert scale > 0.0

    def test_flat_constant_series(self):
        """Test series where all readings are identical."""
        values = np.array([20.0] * 30)
        median, raw_mad, dispersion_name, scale = calculate_robust_baseline(values)

        assert median == 20.0
        assert raw_mad == 0.0
        assert scale == 0.0
        assert "Zero Dispersion" in dispersion_name


class TestIntegrationAndApp:
    """End-to-end integration and Streamlit app tests."""

    def test_real_beijing_prsa_dataset(self):
        """Test loading and analyzing a real Beijing PRSA CSV file if present in workspace."""
        from pathlib import Path
        sample_path = Path("PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv")
        if not sample_path.exists():
            pytest.skip("Real Beijing PRSA dataset not present in workspace.")

        df, report = load_air_quality_csv(sample_path, is_synthetic=False)
        assert report.is_valid is True
        assert report.total_rows > 30000
        assert report.usable_rows > 30000
        assert "Aotizhongxin" in report.available_stations
        assert "PM2.5" in report.available_pollutants

        result = detect_anomalies_mad(df, pollutant="PM2.5", station="Aotizhongxin", k_multiplier=3.0)
        assert result.sufficient_data is True
        assert result.baseline_median is not None
        assert result.threshold_value is not None
        assert result.anomaly_count > 0

    def test_streamlit_app_renders_cleanly(self):
        """Test that app.py initializes and executes without uncaught exceptions using Streamlit AppTest."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py")
        at.run(timeout=15)
        assert not at.exception

