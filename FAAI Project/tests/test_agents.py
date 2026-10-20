"""
Tests for BreatheOps AI Phase 2: Multi-Agent Investigation and Orchestration.

Covers:
1. Successful investigation with a detected anomaly.
2. Successful investigation with no detected anomaly.
3. Missing or unusable data.
4. Insufficient historical records.
5. Cross-station comparison when sufficient records exist.
6. Unsupported conclusions being flagged or qualified by the Critic.
7. Final report containing correct empirical findings.
"""

from io import StringIO
import numpy as np
import pandas as pd
import pytest

from breatheops.agents.base import AgentStatus
from breatheops.agents.critic import CriticSafetyAgent, ValidationStatus
from breatheops.agents.data_quality import DataQualityAgent
from breatheops.agents.multi_station import MultiStationContextAgent
from breatheops.agents.planner import PlannerAgent
from breatheops.agents.response_planner import InvestigationPriority, ResponsePlanningAgent
from breatheops.agents.spike_detection import PollutionSpikeDetectionAgent
from breatheops.demo_data import generate_demo_data
from breatheops.orchestrator import InvestigationContext, InvestigationOrchestrator, InvestigationReport


class TestMultiAgentInvestigation:
    """Test suite for the multi-agent investigation workflow and orchestration."""

    def test_successful_investigation_with_detected_anomaly(self):
        """Verify end-to-end multi-agent execution when an anomalous spike is present."""
        # 14-day demo data with injected spike on Aotizhongxin
        df = generate_demo_data(seed=42, days=14, inject_spike=True, spike_station="Aotizhongxin")
        orchestrator = InvestigationOrchestrator()

        report, context = orchestrator.run_investigation(
            df=df,
            target_station="Aotizhongxin",
            target_pollutant="PM2.5",
            k_multiplier=3.0,
            is_synthetic=True,
            dataset_source_label="Synthetic Demo (Seed 42)",
        )

        assert isinstance(report, InvestigationReport)
        assert context.investigation_id.startswith("INV-")

        # 1. Planner check
        assert orchestrator.planner.name in context.agent_results
        planner_res = context.agent_results[orchestrator.planner.name]
        assert planner_res.status == AgentStatus.COMPLETED
        assert len(context.plan.planned_steps) >= 5

        # 2. Data Quality check
        dq_res = context.agent_results[orchestrator.data_quality.name]
        assert dq_res.status == AgentStatus.COMPLETED
        assert context.data_quality_decision == "CAN_PROCEED"

        # 3. Spike Detection check
        spike_res = context.agent_results[orchestrator.spike_detection.name]
        assert spike_res.status == AgentStatus.COMPLETED
        assert report.anomaly_count > 0
        assert report.peak_reading > 200.0
        assert report.threshold_value is not None

        # 4. Multi-Station check
        multi_res = context.agent_results[orchestrator.multi_station.name]
        assert multi_res.status == AgentStatus.COMPLETED
        assert context.multi_station_evidence.get("cross_station_available") is True
        assert context.multi_station_evidence.get("spatial_classification") in [
            "ISOLATED_LOCAL_ELEVATION", "MODERATE_LOCAL_DISPERSION"
        ]

        # 5. Response Planner check
        resp_res = context.agent_results[orchestrator.response_planner.name]
        assert resp_res.status == AgentStatus.COMPLETED
        assert report.assigned_priority == InvestigationPriority.HIGH_PRIORITY_INVESTIGATION.value
        assert len(report.recommended_actions) >= 2

        # 6. Critic check
        critic_res = context.agent_results[orchestrator.critic.name]
        assert critic_res.status in [AgentStatus.COMPLETED, AgentStatus.WARNING]
        assert report.validation_status in [ValidationStatus.VALIDATED.value, ValidationStatus.REQUIRES_HUMAN_REVIEW.value]

        # 7. Report content check
        assert f"{report.anomaly_count}" in report.markdown_report
        assert "Non-Regulatory Disclaimer" in report.markdown_report

    def test_successful_investigation_with_no_detected_anomaly(self):
        """Verify multi-agent workflow when observations remain nominal."""
        rng = np.random.default_rng(999)
        hours = 96
        timestamps = pd.date_range("2023-01-01 00:00:00", periods=hours, freq="h")
        clean_pm25 = np.clip(30.0 + rng.normal(0, 2.0, hours), 20.0, 40.0)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "station": "Clean_Site",
            "PM2.5": clean_pm25,
            "PM10": clean_pm25 * 1.4,
            "TEMP": 10.0,
            "WSPM": 2.5,
        })

        orchestrator = InvestigationOrchestrator()
        report, context = orchestrator.run_investigation(
            df=df,
            target_station="Clean_Site",
            target_pollutant="PM2.5",
            k_multiplier=3.5,
        )

        assert report.anomaly_count == 0
        assert report.assigned_priority == InvestigationPriority.ROUTINE_MONITORING.value
        assert "no statistical anomalies" in report.plain_language_advisory.lower()
        # Verify it does not falsely claim the air is guaranteed clean/safe
        assert "not a declaration that the air is clean or safe" in report.plain_language_advisory.lower()

    def test_missing_or_unusable_data(self):
        """Test conditional branching when required columns are absent."""
        # DataFrame missing both timestamps and pollutants
        bad_df = pd.DataFrame({
            "TEMP": [15.0, 16.0],
            "RAIN": [0.0, 0.0],
        })

        orchestrator = InvestigationOrchestrator()
        report, context = orchestrator.run_investigation(
            df=bad_df,
            target_station="Default_Station",
            target_pollutant="PM2.5",
        )

        assert context.data_quality_decision == "BLOCKED"
        # Verify conditional skip
        assert context.agent_results[orchestrator.spike_detection.name].status == AgentStatus.SKIPPED
        assert context.agent_results[orchestrator.multi_station.name].status == AgentStatus.SKIPPED
        # Verify response planning issued data audit advisory
        assert report.assigned_priority == InvestigationPriority.DATA_INTEGRITY_AUDIT.value
        assert any("audit" in rec.get("title", "").lower() for rec in report.recommended_actions)

    def test_insufficient_historical_records(self):
        """Test single-row and small-sample dataset limitations handling."""
        single_row_df = pd.DataFrame({
            "timestamp": [pd.to_datetime("2023-01-01 00:00:00")],
            "station": ["Single_Site"],
            "PM2.5": [55.0],
            "PM10": [80.0],
        })

        orchestrator = InvestigationOrchestrator()
        report, context = orchestrator.run_investigation(
            df=single_row_df,
            target_station="Single_Site",
            target_pollutant="PM2.5",
        )

        assert context.data_quality_evidence.get("is_single_row") is True
        assert report.assigned_priority == InvestigationPriority.PROVISIONAL_OBSERVATION.value
        assert any("single row" in lim.lower() or "insufficient" in lim.lower() for lim in report.limitations)
        assert report.anomaly_count == 0

    def test_cross_station_comparison_sufficient_records(self):
        """Test cross-station evaluation when multiple stations are available."""
        df = generate_demo_data(seed=42, days=7, inject_spike=True, spike_station="Aotizhongxin")
        orchestrator = InvestigationOrchestrator()

        report, context = orchestrator.run_investigation(
            df=df,
            target_station="Aotizhongxin",
            target_pollutant="PM2.5",
        )

        multi_ev = context.multi_station_evidence
        assert multi_ev.get("cross_station_available") is True
        assert multi_ev.get("station_count") == 4
        assert len(multi_ev.get("station_window_stats", {})) == 4
        assert "ratio_target_to_others" in multi_ev
        assert report.cross_station_finding != ""

    def test_unsupported_conclusions_flagged_by_critic(self):
        """Test that Critic flags unauthorized causal claims and disproportionate priority."""
        context = InvestigationContext(
            investigation_id="TEST-CRITIC-01",
            target_station="Station_X",
            target_pollutant="PM2.5",
        )
        context.data_quality_evidence = {"decision": "CAN_PROCEED", "is_single_row": False}
        context.spike_evidence = {
            "pollutant": "PM2.5",
            "anomaly_count": 0,  # 0 anomalies!
            "valid_observations": 100,
            "baseline_median": 35.0,
            "threshold_value": 75.0,
            "sufficient_data": True,
        }
        # Inappropriately high priority for 0 anomalies and containing prohibited text
        context.assigned_priority = "HIGH_PRIORITY_INVESTIGATION"
        context.response_evidence = {
            "assigned_priority": "HIGH_PRIORITY_INVESTIGATION",
            "priority_reasoning": "Official emergency declared! Proven that factory caused severe air pollution.",
            "recommendations": [
                {
                    "action_id": "ACT-01",
                    "title": "Legal Action",
                    "target_team": "Enforcement",
                    "urgency": "Emergency",
                    "rationale": "Regulatory violation confirmed.",
                    "details": "Issue fine.",
                }
            ],
        }
        context.multi_station_evidence = {
            "cross_station_available": False,
            "spatial_classification": "UNAVAILABLE",
        }

        critic = CriticSafetyAgent()
        result = critic.execute(context)

        # Critic must flag prohibited phrases and disproportionate priority
        assert result.status == AgentStatus.WARNING
        assert context.validation_status == ValidationStatus.REQUIRES_HUMAN_REVIEW.value
        assert len(result.warnings) > 0
        # Priority should have been auto-corrected from HIGH to ROUTINE
        assert context.assigned_priority == "ROUTINE_MONITORING"
        assert any("prohibited" in w.lower() or "regulatory" in w.lower() for w in result.warnings)

    def test_final_report_contains_correct_findings(self):
        """Verify that the generated markdown report accurately quotes computed statistics."""
        df = generate_demo_data(seed=42, days=7, inject_spike=True, spike_station="Aotizhongxin")
        orchestrator = InvestigationOrchestrator()

        report, context = orchestrator.run_investigation(
            df=df,
            target_station="Aotizhongxin",
            target_pollutant="PM2.5",
        )

        spike_ev = context.spike_evidence

        # Check empirical correspondence
        assert f"{spike_ev['baseline_median']:.2f}" in report.markdown_report
        assert f"{spike_ev['threshold_value']:.2f}" in report.markdown_report
        assert f"{report.anomaly_count}" in report.markdown_report
        assert report.investigation_id in report.markdown_report
        assert report.target_station in report.markdown_report
        assert report.target_pollutant in report.markdown_report
