"""
Response Planning Agent for BreatheOps AI.

Synthesizes empirical findings from prior agents to assign an explainable
investigation priority and formulate proportionate follow-up recommendations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from breatheops.agents.base import AgentResult, AgentStatus, BaseAgent


class InvestigationPriority(str, Enum):
    """Explainable investigation priority ratings (non-regulatory)."""
    ROUTINE_MONITORING = "ROUTINE_MONITORING"
    ELEVATED_INVESTIGATION = "ELEVATED_INVESTIGATION"
    HIGH_PRIORITY_INVESTIGATION = "HIGH_PRIORITY_INVESTIGATION"
    DATA_INTEGRITY_AUDIT = "DATA_INTEGRITY_AUDIT"
    PROVISIONAL_OBSERVATION = "PROVISIONAL_OBSERVATION"


@dataclass
class RecommendedAction:
    """A practical follow-up action with an explicit empirical rationale."""
    action_id: str
    title: str
    target_team: str
    urgency: str
    rationale: str
    details: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "action_id": self.action_id,
            "title": self.title,
            "target_team": self.target_team,
            "urgency": self.urgency,
            "rationale": self.rationale,
            "details": self.details,
        }


class ResponsePlanningAgent(BaseAgent):
    """Response Planning Agent assigning transparent priorities and action steps."""

    def __init__(self):
        super().__init__(
            name="Response Planning Agent",
            description="Assigns explainable investigation priority and recommends proportionate next steps.",
        )

    def execute(self, context: Any) -> AgentResult:
        """Formulate priority and actionable recommendations based on agent findings."""
        dq_ev: Dict[str, Any] = getattr(context, "data_quality_evidence", {})
        spike_ev: Dict[str, Any] = getattr(context, "spike_evidence", {})
        multi_ev: Dict[str, Any] = getattr(context, "multi_station_evidence", {})
        target_station: str = getattr(context, "target_station", "All")
        target_pollutant: str = getattr(context, "target_pollutant", "PM2.5")

        dq_decision = dq_ev.get("decision", "CAN_PROCEED")
        is_single_row = dq_ev.get("is_single_row", False)
        anomalies_detected = spike_ev.get("anomalies_detected", False)
        anomaly_count = spike_ev.get("anomaly_count", 0)
        peak_ratio = spike_ev.get("peak_ratio_to_median", 1.0)
        sufficient_data = spike_ev.get("sufficient_data", True)
        spatial_class = multi_ev.get("spatial_classification", "UNAVAILABLE_SINGLE_STATION")
        weather_ev = multi_ev.get("weather_context", {})

        recommendations: List[RecommendedAction] = []
        action_counter = 1

        # 1. Determine priority using transparent decision matrix
        if dq_decision == "BLOCKED":
            priority = InvestigationPriority.DATA_INTEGRITY_AUDIT
            priority_reason = (
                "Data quality gate failed due to missing required columns, empty records, or unrecoverable formatting errors. "
                "Investigation cannot evaluate environmental concentrations until data integrity is resolved."
            )
            recommendations.append(
                RecommendedAction(
                    action_id=f"ACT-{action_counter:02d}",
                    title="Audit Data Ingestion Pipeline & File Headers",
                    target_team="Data Engineering / Data Custodian",
                    urgency="Immediate",
                    rationale="Required temporal or pollutant variables could not be parsed.",
                    details="Review CSV source format, timestamp encoding, and column naming to conform with standard schemas.",
                )
            )
            action_counter += 1

        elif is_single_row or not sufficient_data:
            priority = InvestigationPriority.PROVISIONAL_OBSERVATION
            priority_reason = (
                f"Historical baseline is statistically insufficient ({dq_ev.get('valid_target_readings', 0)} readings). "
                "Observations cannot be reliably categorized without a historical baseline."
            )
            recommendations.append(
                RecommendedAction(
                    action_id=f"ACT-{action_counter:02d}",
                    title="Expand Continuous Time-Series Sampling",
                    target_team="Environmental Monitoring Station Operations",
                    urgency="Standard",
                    rationale="At least 24 hours (preferably 14+ days) of hourly observations are required to compute robust median and MAD baselines.",
                    details=f"Deploy automated continuous hourly logging at station '{target_station}' before concluding trend analyses.",
                )
            )
            action_counter += 1

        elif not anomalies_detected or anomaly_count == 0:
            priority = InvestigationPriority.ROUTINE_MONITORING
            priority_reason = (
                f"All {spike_ev.get('valid_observations', 0)} hourly observations for {target_pollutant} remained within "
                f"historical baseline bounds (Peak: {spike_ev.get('max_observed', 0.0)} vs Threshold: {spike_ev.get('threshold_value', 0.0)}). "
                "Standard background monitoring is sufficient."
            )
            recommendations.append(
                RecommendedAction(
                    action_id=f"ACT-{action_counter:02d}",
                    title="Maintain Baseline Monitoring Schedule",
                    target_team="Automated Monitoring Team",
                    urgency="Routine",
                    rationale="No statistical deviations detected above the 3.0× MAD baseline threshold.",
                    details=f"Continue regular hourly telemetry for station '{target_station}' without special intervention.",
                )
            )
            action_counter += 1

        elif anomaly_count > 5 or peak_ratio >= 2.5:
            priority = InvestigationPriority.HIGH_PRIORITY_INVESTIGATION
            priority_reason = (
                f"Significant statistical event: {anomaly_count} anomalous readings detected, peaking at {peak_ratio:.1f}× "
                f"above station median ({spike_ev.get('peak_anomaly_value', 0.0)} {spike_ev.get('unit', '')}). "
                "Confidence is supported by sufficient historical baseline data."
            )

            # Recommendations based on spatial findings
            if spatial_class == "ISOLATED_LOCAL_ELEVATION":
                recommendations.append(
                    RecommendedAction(
                        action_id=f"ACT-{action_counter:02d}",
                        title="Perform Sensor Calibration and Local Inlet Inspection",
                        target_team="Station Field Operations & QA/QC",
                        urgency="High",
                        rationale=f"The spike was isolated to station '{target_station}', while other stations remained nominal. "
                                  "Hardware drift, inlet obstruction, or immediate micro-environment sources must be audited first.",
                        details=f"Inspect optical/beta-attenuation sensor optics, flow rate, and inlet lines at station '{target_station}'.",
                    )
                )
                action_counter += 1
            else:
                recommendations.append(
                    RecommendedAction(
                        action_id=f"ACT-{action_counter:02d}",
                        title="Cross-Station Regional Correlation Review",
                        target_team="Regional Air Quality Analysis Bureau",
                        urgency="High",
                        rationale="Concurrent elevation observed across multiple stations suggests wider dispersion rather than single-sensor error.",
                        details="Integrate data from adjacent monitoring stations and municipal networks to trace event boundary extent.",
                    )
                )
                action_counter += 1

            if weather_ev and weather_ev.get("avg_wind_speed_m_s", 5.0) < 1.5:
                recommendations.append(
                    RecommendedAction(
                        action_id=f"ACT-{action_counter:02d}",
                        title="Examine Boundary-Layer Meteorological Profiles",
                        target_team="Atmospheric Modeling Group",
                        urgency="Medium",
                        rationale=f"Low wind speeds ({weather_ev.get('avg_wind_speed_m_s')} m/s) coincide with the spike window, which may inhibit vertical mixing.",
                        details="Review boundary layer height, temperature inversion soundings, and atmospheric stability indices.",
                    )
                )
                action_counter += 1

            recommendations.append(
                RecommendedAction(
                    action_id=f"ACT-{action_counter:02d}",
                    title="Escalate Dossier for Certified Human Expert Review",
                    target_team="Senior Environmental Scientist / Panel",
                    urgency="High",
                    rationale="High magnitude anomaly confirmed by statistical metrics requires expert verification before publication.",
                    details="Forward this automated investigation dossier with raw sensor logs to senior staff for validation.",
                )
            )
            action_counter += 1

        else:
            # Moderate anomaly count / ratio
            priority = InvestigationPriority.ELEVATED_INVESTIGATION
            priority_reason = (
                f"Moderate statistical deviation: {anomaly_count} reading(s) exceeded the threshold with peak "
                f"{peak_ratio:.1f}× baseline. Warrants verification without immediate crisis intervention."
            )
            recommendations.append(
                RecommendedAction(
                    action_id=f"ACT-{action_counter:02d}",
                    title="Verify Telemetry and Flag Potential Local Transient Source",
                    target_team="Air Quality Monitoring Analysts",
                    urgency="Standard",
                    rationale=f"Observed {anomaly_count} transient spike(s) at '{target_station}'.",
                    details="Check station telemetry for transmission glitches, power interruptions, or nearby short-term activity.",
                )
            )
            action_counter += 1

        evidence: Dict[str, Any] = {
            "assigned_priority": priority.value,
            "priority_reasoning": priority_reason,
            "recommendation_count": len(recommendations),
            "recommendations": [r.to_dict() for r in recommendations],
            "non_regulatory_rule": "Statistical anomaly status is not an official regulatory violation or AQI emergency declaration.",
        }

        context.response_evidence = evidence
        context.assigned_priority = priority.value

        summary_msg = (
            f"Assigned priority [{priority.value}]. Reason: {priority_reason} "
            f"Formulated {len(recommendations)} actionable recommendation(s)."
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.COMPLETED,
            summary=summary_msg,
            evidence=evidence,
            next_action="Invoke Critic and Safety Agent to validate evidence integrity and proportional recommendations.",
        )
