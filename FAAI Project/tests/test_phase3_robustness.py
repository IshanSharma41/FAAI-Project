"""
Phase 3 Robustness, Error Handling, and Usability Tests for BreatheOps AI.

Covers:
1. Plain text report generation and empirical consistency.
2. Handling datasets missing all weather columns.
3. Handling datasets missing station column.
4. Handling corrupted / malformed CSV values and mixed datatypes.
5. Handling completely empty CSV buffers.
6. Dual-format report agreement (Markdown vs Plain Text).
7. Streamlit headless execution of Phase 3 interface with AppTest.
"""

from io import StringIO
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from breatheops.data_loader import load_air_quality_csv
from breatheops.demo_data import generate_demo_data
from breatheops.orchestrator import InvestigationOrchestrator, InvestigationReport


class TestPhase3Robustness:
    """Robustness and error handling test suite for Phase 3."""

    def test_plain_text_report_generation(self):
        """Verify plain text report is cleanly generated with required sections."""
        df = generate_demo_data(seed=42, days=7, inject_spike=True, spike_station="Aotizhongxin")
        orchestrator = InvestigationOrchestrator()
        report, context = orchestrator.run_investigation(
            df=df,
            target_station="Aotizhongxin",
            target_pollutant="PM2.5",
        )

        assert isinstance(report.plain_text_report, str)
        assert len(report.plain_text_report) > 200
        assert report.investigation_id in report.plain_text_report
        assert "BREATHEOPS AI" in report.plain_text_report
        assert "1. EXECUTIVE ADVISORY & SUMMARY" in report.plain_text_report
        assert "2. STATISTICAL FINDINGS & EMPIRICAL BASELINE" in report.plain_text_report
        assert "3. MULTI-STATION & METEOROLOGICAL CONTEXT" in report.plain_text_report
        assert "4. RECOMMENDED ACTION PLAN" in report.plain_text_report
        assert "5. VERIFICATION & SAFETY REVIEW" in report.plain_text_report
        assert "ACADEMIC & NON-REGULATORY DISCLAIMER" in report.plain_text_report

    def test_missing_weather_columns_in_orchestration(self):
        """Test investigation pipeline on dataset containing pollutants only, with no weather columns."""
        timestamps = pd.date_range("2023-01-01 00:00:00", periods=48, freq="h")
        df_no_weather = pd.DataFrame({
            "timestamp": timestamps,
            "station": "Station_No_Weather",
            "PM2.5": np.random.default_rng(42).normal(40, 5, 48),
            "PM10": np.random.default_rng(42).normal(60, 8, 48),
        })

        orchestrator = InvestigationOrchestrator()
        report, context = orchestrator.run_investigation(
            df=df_no_weather,
            target_station="Station_No_Weather",
            target_pollutant="PM2.5",
        )

        assert report is not None
        assert context.data_quality_decision == "CAN_PROCEED"
        assert context.multi_station_evidence.get("weather_available") is False
        assert "No meteorological sensor columns" in report.weather_context or "evaluated" in report.weather_context

    def test_missing_station_column_in_orchestration(self):
        """Test CSV without station column is assigned Default_Station and analyzes cleanly."""
        csv_data = """timestamp, PM2.5, PM10
2023-01-01 00:00:00, 35.0, 50.0
2023-01-01 01:00:00, 38.0, 52.0
2023-01-01 02:00:00, 42.0, 58.0
2023-01-01 03:00:00, 45.0, 60.0
"""
        df, report = load_air_quality_csv(StringIO(csv_data))
        assert report.is_valid is True
        assert "Default_Station" in report.available_stations

        orchestrator = InvestigationOrchestrator()
        inv_report, context = orchestrator.run_investigation(
            df=df,
            target_station="Default_Station",
            target_pollutant="PM2.5",
            validation_report=report,
        )
        assert inv_report.target_station == "Default_Station"

    def test_corrupted_csv_strings_and_empty_rows(self):
        """Test safe handling of non-numeric garbage in pollutant columns."""
        csv_data = """timestamp, station, PM2.5, PM10
2023-01-01 00:00:00, Site_A, 30.5, 45.0
2023-01-01 01:00:00, Site_A, CORRUPTED_DATA, 48.0
2023-01-01 02:00:00, Site_A, ???, 50.0
2023-01-01 03:00:00, Site_A, 33.2, NULL
"""
        df, report = load_air_quality_csv(StringIO(csv_data))
        assert report.is_valid is True
        # Two invalid values should be safely coerced to NaN
        assert report.missing_pollutant_counts["PM2.5"] == 2
        assert pd.isna(df.loc[1, "PM2.5"])
        assert pd.isna(df.loc[2, "PM2.5"])

    def test_completely_empty_csv_file(self):
        """Test uploading a zero-byte empty file."""
        empty_buffer = StringIO("")
        df, report = load_air_quality_csv(empty_buffer)
        assert df is None
        assert report.is_valid is False
        assert any("empty" in err.lower() or "0 rows" in err.lower() for err in report.errors)

    def test_markdown_and_plain_text_consistency(self):
        """Ensure both Markdown and Plain Text reports contain identical computed values."""
        df = generate_demo_data(seed=42, days=7, inject_spike=True, spike_station="Aotizhongxin")
        orchestrator = InvestigationOrchestrator()
        report, context = orchestrator.run_investigation(
            df=df,
            target_station="Aotizhongxin",
            target_pollutant="PM2.5",
        )

        md = report.markdown_report
        txt = report.plain_text_report

        # Both must contain investigation ID, priority, validation status, and anomaly count
        assert report.investigation_id in md and report.investigation_id in txt
        assert report.assigned_priority in md and report.assigned_priority in txt
        assert report.validation_status in md and report.validation_status in txt
        assert f"{report.anomaly_count}" in md and f"{report.anomaly_count}" in txt

    def test_phase3_streamlit_app_renders_with_apptest(self):
        """Headless execution verifying Phase 3 Streamlit interface runs end-to-end."""
        at = AppTest.from_file("app.py")
        at.run(timeout=15)
        assert not at.exception
