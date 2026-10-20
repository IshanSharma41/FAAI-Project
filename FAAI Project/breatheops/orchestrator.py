"""
Multi-Agent Orchestrator and Evidence-Based Report Generator for BreatheOps AI.

Coordinates the 6 specialized agents, passes structured outputs across the workflow,
executes conditional branching, and synthesizes the finalized Investigation Report.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from breatheops.agents.base import AgentResult, AgentStatus
from breatheops.agents.critic import CriticSafetyAgent, ValidationStatus
from breatheops.agents.data_quality import DataQualityAgent
from breatheops.agents.multi_station import MultiStationContextAgent
from breatheops.agents.planner import InvestigationPlan, PlannerAgent
from breatheops.agents.response_planner import InvestigationPriority, ResponsePlanningAgent
from breatheops.agents.spike_detection import PollutionSpikeDetectionAgent
from breatheops.analysis import AnomalyDetectionResult, POLLUTANT_UNITS
from breatheops.data_loader import DataValidationReport


@dataclass
class InvestigationContext:
    """Shared state passed between specialized agents during an investigation."""
    investigation_id: str
    target_station: str
    target_pollutant: str
    k_multiplier: float = 3.0
    is_synthetic: bool = False
    dataset_source_label: str = "Direct Input"

    # Datasets
    raw_df: Optional[pd.DataFrame] = None
    cleaned_df: Optional[pd.DataFrame] = None
    validation_report: Optional[DataValidationReport] = None

    # Agent Artifacts
    plan: Optional[InvestigationPlan] = None
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)

    # Structured Evidence
    data_quality_decision: str = "PENDING"
    data_quality_evidence: Dict[str, Any] = field(default_factory=dict)

    spike_detection_result: Optional[AnomalyDetectionResult] = None
    spike_evidence: Dict[str, Any] = field(default_factory=dict)

    multi_station_evidence: Dict[str, Any] = field(default_factory=dict)

    assigned_priority: str = "ROUTINE_MONITORING"
    response_evidence: Dict[str, Any] = field(default_factory=dict)

    validation_status: str = "PENDING"
    critic_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvestigationReport:
    """Complete, evidence-grounded report produced by the multi-agent investigation."""
    investigation_id: str
    generated_at: str
    dataset_source: str
    observation_period: str
    target_station: str
    target_pollutant: str
    validation_status: str
    assigned_priority: str
    priority_reasoning: str
    plain_language_advisory: str
    data_quality_findings: str
    statistical_method: str
    baseline_median: Optional[float]
    threshold_value: Optional[float]
    anomaly_count: int
    peak_reading: Optional[float]
    cross_station_finding: str
    weather_context: str
    recommended_actions: List[Dict[str, str]]
    limitations: List[str]
    critic_audit_summary: str
    agent_execution_log: List[Dict[str, Any]]
    markdown_report: str = ""
    plain_text_report: str = ""


class InvestigationOrchestrator:
    """Dedicated orchestrator coordinating all 6 specialized agents."""

    def __init__(self):
        self.planner = PlannerAgent()
        self.data_quality = DataQualityAgent()
        self.spike_detection = PollutionSpikeDetectionAgent()
        self.multi_station = MultiStationContextAgent()
        self.response_planner = ResponsePlanningAgent()
        self.critic = CriticSafetyAgent()

    def run_investigation(
        self,
        df: pd.DataFrame,
        target_station: str,
        target_pollutant: str,
        k_multiplier: float = 3.0,
        is_synthetic: bool = False,
        dataset_source_label: str = "Input Data",
        validation_report: Optional[DataValidationReport] = None,
    ) -> tuple[InvestigationReport, InvestigationContext]:
        """Execute the coordinated 6-agent investigation pipeline."""
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        inv_id = f"INV-{now_str}-{abs(hash(target_station + target_pollutant)) % 10000:04d}"

        context = InvestigationContext(
            investigation_id=inv_id,
            target_station=target_station,
            target_pollutant=target_pollutant,
            k_multiplier=k_multiplier,
            is_synthetic=is_synthetic,
            dataset_source_label=dataset_source_label,
            raw_df=df,
            cleaned_df=df if validation_report and validation_report.is_valid else None,
            validation_report=validation_report,
        )

        execution_log: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # Stage 1: Planner Agent
        # -------------------------------------------------------------
        plan_res = self.planner.execute(context)
        context.agent_results[self.planner.name] = plan_res
        execution_log.append(plan_res.to_dict())

        # -------------------------------------------------------------
        # Stage 2: Data Quality Agent
        # -------------------------------------------------------------
        dq_res = self.data_quality.execute(context)
        context.agent_results[self.data_quality.name] = dq_res
        execution_log.append(dq_res.to_dict())
        self._update_plan_step(context, "Data Quality Agent", dq_res.status)

        # Conditional Branch: If Data Quality is BLOCKED
        can_proceed_analytically = context.data_quality_decision != "BLOCKED"

        # -------------------------------------------------------------
        # Stage 3: Pollution Spike Detection Agent
        # -------------------------------------------------------------
        if can_proceed_analytically:
            spike_res = self.spike_detection.execute(context)
            context.agent_results[self.spike_detection.name] = spike_res
            execution_log.append(spike_res.to_dict())
            self._update_plan_step(context, "Pollution Spike Detection Agent", spike_res.status)
        else:
            spike_res = AgentResult(
                agent_name=self.spike_detection.name,
                status=AgentStatus.SKIPPED,
                summary="Spike detection omitted due to failed Data Quality gate.",
                limitations=["Statistical detection blocked by data issues."],
                next_action="Halt spatial analysis.",
            )
            context.agent_results[self.spike_detection.name] = spike_res
            execution_log.append(spike_res.to_dict())
            self._update_plan_step(context, "Pollution Spike Detection Agent", AgentStatus.SKIPPED)

        # -------------------------------------------------------------
        # Stage 4: Multi-Station Context Agent
        # -------------------------------------------------------------
        if can_proceed_analytically:
            multi_res = self.multi_station.execute(context)
            context.agent_results[self.multi_station.name] = multi_res
            execution_log.append(multi_res.to_dict())
            self._update_plan_step(context, "Multi-Station Context Agent", multi_res.status)
        else:
            multi_res = AgentResult(
                agent_name=self.multi_station.name,
                status=AgentStatus.SKIPPED,
                summary="Multi-station context omitted due to data quality block.",
                limitations=["No data available for spatial comparison."],
                next_action="Proceed to Response Planning.",
            )
            context.agent_results[self.multi_station.name] = multi_res
            execution_log.append(multi_res.to_dict())
            self._update_plan_step(context, "Multi-Station Context Agent", AgentStatus.SKIPPED)

        # -------------------------------------------------------------
        # Stage 5: Response Planning Agent
        # -------------------------------------------------------------
        resp_res = self.response_planner.execute(context)
        context.agent_results[self.response_planner.name] = resp_res
        execution_log.append(resp_res.to_dict())
        self._update_plan_step(context, "Response Planning Agent", resp_res.status)

        # -------------------------------------------------------------
        # Stage 6: Critic and Safety Agent
        # -------------------------------------------------------------
        critic_res = self.critic.execute(context)
        context.agent_results[self.critic.name] = critic_res
        execution_log.append(critic_res.to_dict())
        self._update_plan_step(context, "Critic and Safety Agent", critic_res.status)

        # -------------------------------------------------------------
        # Synthesis: Compile Final Evidence Dossier & Report
        # -------------------------------------------------------------
        report = self._compile_report(context, execution_log)
        return report, context

    def _update_plan_step(self, context: InvestigationContext, agent_name: str, status: AgentStatus) -> None:
        """Update the planned step status in the context plan."""
        if context.plan:
            for step in context.plan.planned_steps:
                if step.agent_name == agent_name:
                    step.status = status

    def _compile_report(
        self,
        context: InvestigationContext,
        execution_log: List[Dict[str, Any]],
    ) -> InvestigationReport:
        """Assemble all agent findings into a structured InvestigationReport and Markdown document."""
        dq_ev = context.data_quality_evidence
        spike_ev = context.spike_evidence
        multi_ev = context.multi_station_evidence
        resp_ev = context.response_evidence
        critic_ev = context.critic_evidence

        unit = POLLUTANT_UNITS.get(context.target_pollutant, "µg/m³")

        # Temporal bounds
        if context.cleaned_df is not None and not context.cleaned_df.empty and "timestamp" in context.cleaned_df.columns:
            min_ts = str(context.cleaned_df["timestamp"].min())
            max_ts = str(context.cleaned_df["timestamp"].max())
            obs_period = f"{min_ts} to {max_ts}"
        else:
            obs_period = "Unspecified / Incomplete Temporal Records"

        # Limitations gathered from all agents
        all_limitations: List[str] = []
        for res in context.agent_results.values():
            all_limitations.extend(res.limitations)
        # Deduplicate while preserving order
        unique_limitations = list(dict.fromkeys(all_limitations))

        # Plain language advisory
        if context.spike_detection_result:
            plain_advisory = context.spike_detection_result.plain_language_finding
        else:
            plain_advisory = resp_ev.get("priority_reasoning", "Investigation completed.")

        report = InvestigationReport(
            investigation_id=context.investigation_id,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            dataset_source=context.dataset_source_label,
            observation_period=obs_period,
            target_station=context.target_station,
            target_pollutant=context.target_pollutant,
            validation_status=context.validation_status,
            assigned_priority=context.assigned_priority,
            priority_reasoning=resp_ev.get("priority_reasoning", ""),
            plain_language_advisory=plain_advisory,
            data_quality_findings=context.agent_results.get(self.data_quality.name, AgentResult("", AgentStatus.IDLE, "")).summary,
            statistical_method=(
                f"Median + {context.k_multiplier}× MAD ({spike_ev.get('dispersion_metric', 'MAD')})"
                if spike_ev else "N/A"
            ),
            baseline_median=spike_ev.get("baseline_median"),
            threshold_value=spike_ev.get("threshold_value"),
            anomaly_count=spike_ev.get("anomaly_count", 0),
            peak_reading=spike_ev.get("peak_anomaly_value") or spike_ev.get("max_observed"),
            cross_station_finding=multi_ev.get("spatial_description", "Cross-station comparison was not available."),
            weather_context=multi_ev.get("weather_summary") or "Meteorological data evaluated.",
            recommended_actions=resp_ev.get("recommendations", []),
            limitations=unique_limitations,
            critic_audit_summary=context.agent_results.get(self.critic.name, AgentResult("", AgentStatus.IDLE, "")).summary,
            agent_execution_log=execution_log,
        )

        report.markdown_report = self._build_markdown(report, unit)
        report.plain_text_report = self._build_plain_text(report, unit)
        return report

    def _build_markdown(self, report: InvestigationReport, unit: str) -> str:
        """Render a formatted, comprehensive markdown report."""
        lines = [
            f"# BreatheOps AI — Multi-Agent Investigation Report",
            f"**Investigation ID**: `{report.investigation_id}`  ",
            f"**Generated**: {report.generated_at}  ",
            f"**Dataset Source**: {report.dataset_source}  ",
            f"**Monitoring Target**: Station `{report.target_station}` | Pollutant `{report.target_pollutant}`  ",
            f"**Observation Period**: {report.observation_period}  ",
            "",
            "---",
            "",
            "## 1. Executive Summary & Advisory",
            f"> **Investigation Priority**: `{report.assigned_priority}`  ",
            f"> **Validation Status**: `{report.validation_status}`  ",
            "",
            f"{report.plain_language_advisory}",
            "",
            "---",
            "",
            "## 2. Statistical Findings & Empirical Evidence",
            "| Metric | Empirical Value |",
            "| :--- | :--- |",
            f"| **Station Baseline (Median)** | {f'{report.baseline_median:.2f} {unit}' if report.baseline_median is not None else 'N/A'} |",
            f"| **Detection Threshold** | {f'{report.threshold_value:.2f} {unit}' if report.threshold_value is not None else 'N/A'} |",
            f"| **Statistical Rule** | {report.statistical_method} |",
            f"| **Anomalous Observations Detected** | **{report.anomaly_count}** |",
            f"| **Peak Observed Reading** | {f'{report.peak_reading:.2f} {unit}' if report.peak_reading is not None else 'N/A'} |",
            "",
            "---",
            "",
            "## 3. Multi-Station & Meteorological Context",
            f"- **Spatial Findings**: {report.cross_station_finding}",
            "",
            "---",
            "",
            "## 4. Recommended Action Plan",
        ]

        if report.recommended_actions:
            for rec in report.recommended_actions:
                lines.append(f"### [{rec.get('action_id')}] {rec.get('title')}")
                lines.append(f"- **Target Team**: {rec.get('target_team')}")
                lines.append(f"- **Urgency**: {rec.get('urgency')}")
                lines.append(f"- **Rationale**: {rec.get('rationale')}")
                lines.append(f"- **Action Details**: {rec.get('details')}")
                lines.append("")
        else:
            lines.append("No specialized follow-up actions required; continue routine monitoring.")

        lines.extend([
            "---",
            "",
            "## 5. Verification & Safety Review (Critic Agent)",
            f"{report.critic_audit_summary}",
            "",
            "### Investigation Limitations & Uncertainty Bounds",
        ])

        for lim in report.limitations:
            lines.append(f"- ⚠️ {lim}")

        lines.extend([
            "",
            "> ℹ️ **Mandatory Academic & Non-Regulatory Disclaimer**:",
            "> Statistical anomaly detection identifies measurements that deviate significantly from station baseline norms. "
            "> This analysis does not represent an official regulatory violation, statutory compliance determination, "
            "> or official environmental emergency classification.",
        ])

        return "\n".join(lines)

    def _build_plain_text(self, report: InvestigationReport, unit: str) -> str:
        """Render a formatted, human-readable plain text investigation report."""
        sep = "=" * 78
        sub_sep = "-" * 78
        lines = [
            sep,
            "BREATHEOPS AI — MULTI-AGENT AIR QUALITY INVESTIGATION REPORT",
            sep,
            f"Investigation ID     : {report.investigation_id}",
            f"Generated Timestamp  : {report.generated_at}",
            f"Dataset Source       : {report.dataset_source}",
            f"Target Station       : {report.target_station}",
            f"Target Pollutant     : {report.target_pollutant}",
            f"Observation Window   : {report.observation_period}",
            f"Investigation Status : Priority [{report.assigned_priority}] | Certification [{report.validation_status}]",
            sub_sep,
            "",
            "1. EXECUTIVE ADVISORY & SUMMARY",
            sub_sep,
            report.plain_language_advisory,
            "",
            "2. STATISTICAL FINDINGS & EMPIRICAL BASELINE",
            sub_sep,
            f"Station Baseline (Median) : {f'{report.baseline_median:.2f} {unit}' if report.baseline_median is not None else 'N/A'}",
            f"Detection Threshold       : {f'{report.threshold_value:.2f} {unit}' if report.threshold_value is not None else 'N/A'}",
            f"Statistical Rule Applied  : {report.statistical_method}",
            f"Anomalous Points Detected : {report.anomaly_count}",
            f"Peak Observed Reading     : {f'{report.peak_reading:.2f} {unit}' if report.peak_reading is not None else 'N/A'}",
            "",
            "3. MULTI-STATION & METEOROLOGICAL CONTEXT",
            sub_sep,
            f"Spatial Context : {report.cross_station_finding}",
            f"Weather Context : {report.weather_context}",
            "",
            "4. RECOMMENDED ACTION PLAN",
            sub_sep,
        ]

        if report.recommended_actions:
            for rec in report.recommended_actions:
                lines.append(f"[{rec.get('action_id')}] {rec.get('title')}")
                lines.append(f"  * Target Team  : {rec.get('target_team')}")
                lines.append(f"  * Urgency      : {rec.get('urgency')}")
                lines.append(f"  * Rationale    : {rec.get('rationale')}")
                lines.append(f"  * Action Steps : {rec.get('details')}")
                lines.append("")
        else:
            lines.append("No specialized follow-up actions required; continue routine monitoring.\n")

        lines.extend([
            "5. VERIFICATION & SAFETY REVIEW (CRITIC AGENT)",
            sub_sep,
            report.critic_audit_summary,
            "",
            "Investigation Limitations & Uncertainty Bounds:",
        ])

        for lim in report.limitations:
            lines.append(f"  * WARNING: {lim}")

        lines.extend([
            "",
            sep,
            "ACADEMIC & NON-REGULATORY DISCLAIMER",
            sep,
            "Statistical anomaly detection identifies measurements that deviate significantly",
            "from station baseline norms. This analysis does not represent an official regulatory",
            "violation, statutory air-quality standard compliance determination, or official",
            "environmental emergency classification.",
            sep,
        ])

        return "\n".join(lines)
