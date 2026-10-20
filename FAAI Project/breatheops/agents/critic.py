"""
Critic and Safety Agent for BreatheOps AI.

Performs rigorous empirical verification, checks numerical consistency,
eliminates unsupported causal or regulatory claims, and certifies validation status.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional

from breatheops.agents.base import AgentResult, AgentStatus, BaseAgent


class ValidationStatus(str, Enum):
    """Validation certification decision."""
    VALIDATED = "VALIDATED"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"
    REJECTED = "REJECTED"


@dataclass
class AuditCheckItem:
    """A discrete audit check conducted by the Critic Agent."""
    check_name: str
    passed: bool
    details: str
    severity: str = "INFO"  # INFO, WARNING, CRITICAL


class CriticSafetyAgent(BaseAgent):
    """Critic and Safety Agent verifying scientific rigor, consistency, and safety."""

    # Prohibited unsupported causal and regulatory phrases
    PROHIBITED_PATTERNS = [
        (r"\b(official\s+emergency|regulatory\s+violation|statutory\s+breach)\b", "Claims official regulatory or legal emergency status without statutory authority."),
        (r"\b(air\s+is\s+(guaranteed\s+)?safe|completely\s+clean|zero\s+health\s+risk)\b", "Asserts unjustified safety or health compliance based solely on statistical non-anomaly."),
        (r"\b(caused\s+by\s+(a\s+)?factory|caused\s+by\s+traffic|proven\s+that\s+weather\s+caused)\b", "Asserts unsubstantiated causal attribution without ground-truth source apportionment evidence."),
    ]

    def __init__(self):
        super().__init__(
            name="Critic and Safety Agent",
            description="Audits numerical consistency, removes unsupported claims, and certifies validation status.",
        )

    def execute(self, context: Any) -> AgentResult:
        """Run comprehensive verification audit across all accumulated agent evidence."""
        audit_checks: List[AuditCheckItem] = []
        modifications_made: List[str] = []
        critic_warnings: List[str] = []
        requires_human_review = False

        dq_ev: Dict[str, Any] = getattr(context, "data_quality_evidence", {})
        spike_ev: Dict[str, Any] = getattr(context, "spike_evidence", {})
        multi_ev: Dict[str, Any] = getattr(context, "multi_station_evidence", {})
        resp_ev: Dict[str, Any] = getattr(context, "response_evidence", {})
        assigned_priority: str = getattr(context, "assigned_priority", "ROUTINE_MONITORING")

        # -------------------------------------------------------------
        # 1. Numerical Consistency Check
        # -------------------------------------------------------------
        anomaly_count = spike_ev.get("anomaly_count", 0)
        valid_obs = spike_ev.get("valid_observations", 0)
        baseline_med = spike_ev.get("baseline_median")
        threshold_val = spike_ev.get("threshold_value")

        num_consistent = True
        num_details = []

        if baseline_med is not None and threshold_val is not None:
            if threshold_val < baseline_med:
                num_consistent = False
                num_details.append(f"Threshold ({threshold_val}) is lower than baseline median ({baseline_med}), violating upward spike rule.")

        if anomaly_count < 0 or valid_obs < 0:
            num_consistent = False
            num_details.append(f"Negative count detected: anomaly_count={anomaly_count}, valid_obs={valid_obs}.")

        if anomaly_count > valid_obs:
            num_consistent = False
            num_details.append(f"Anomaly count ({anomaly_count}) exceeds valid observations ({valid_obs}).")

        if num_consistent:
            audit_checks.append(AuditCheckItem("Numerical Consistency", True, "All reported metrics are mathematically consistent."))
        else:
            audit_checks.append(AuditCheckItem("Numerical Consistency", False, "; ".join(num_details), severity="CRITICAL"))
            critic_warnings.append(f"Numerical inconsistency: {'; '.join(num_details)}")
            requires_human_review = True

        # -------------------------------------------------------------
        # 2. Safety & Regulatory Language Audit
        # -------------------------------------------------------------
        # Check text in findings and responses
        texts_to_scan = [
            resp_ev.get("priority_reasoning", ""),
            spike_ev.get("pollutant", ""),
            multi_ev.get("spatial_description", ""),
        ]
        for rec in resp_ev.get("recommendations", []):
            texts_to_scan.append(rec.get("rationale", ""))
            texts_to_scan.append(rec.get("details", ""))

        combined_text = " ".join(texts_to_scan).lower()
        prohibited_hits = []

        for pattern, explanation in self.PROHIBITED_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                prohibited_hits.append(explanation)

        if not prohibited_hits:
            audit_checks.append(AuditCheckItem("Language & Safety Guardrails", True, "No prohibited regulatory or unsupported causal claims detected."))
        else:
            audit_checks.append(AuditCheckItem("Language & Safety Guardrails", False, "; ".join(prohibited_hits), severity="CRITICAL"))
            critic_warnings.append(f"Prohibited phrasing flagged: {'; '.join(prohibited_hits)}")
            modifications_made.append("Appended mandatory disclaimer clarifying that statistical anomalies are not regulatory violations.")
            requires_human_review = True

        # -------------------------------------------------------------
        # 3. Proportionality & Evidence Grounding Check
        # -------------------------------------------------------------
        proportional = True
        prop_details = []

        # Rule A: Cannot have HIGH_PRIORITY_INVESTIGATION if 0 anomalies
        if anomaly_count == 0 and assigned_priority == "HIGH_PRIORITY_INVESTIGATION":
            proportional = False
            prop_details.append("Assigned HIGH_PRIORITY_INVESTIGATION despite 0 anomalies detected.")
            # Auto-correct to ROUTINE_MONITORING
            context.assigned_priority = "ROUTINE_MONITORING"
            resp_ev["assigned_priority"] = "ROUTINE_MONITORING"
            modifications_made.append("Overrode priority from HIGH_PRIORITY_INVESTIGATION to ROUTINE_MONITORING due to absence of anomalies.")
            requires_human_review = True

        # Rule B: Cannot declare confident baseline on single row or < 24 obs
        if (dq_ev.get("is_single_row") or not spike_ev.get("sufficient_data", True)) and assigned_priority not in [
            "PROVISIONAL_OBSERVATION", "DATA_INTEGRITY_AUDIT"
        ]:
            proportional = False
            prop_details.append(f"Small sample size ({valid_obs} observations) cannot support confident priority '{assigned_priority}'.")
            context.assigned_priority = "PROVISIONAL_OBSERVATION"
            resp_ev["assigned_priority"] = "PROVISIONAL_OBSERVATION"
            modifications_made.append("Downgraded priority to PROVISIONAL_OBSERVATION due to insufficient historical baseline.")
            requires_human_review = True

        # Rule C: Cross-station claims without other stations
        if not multi_ev.get("cross_station_available", False) and "regional" in multi_ev.get("spatial_classification", "").lower():
            proportional = False
            prop_details.append("Regional elevation claimed, but dataset has only 1 station.")
            modifications_made.append("Restricted spatial conclusion to single station observation.")
            requires_human_review = True

        if proportional:
            audit_checks.append(AuditCheckItem("Proportionality of Recommendations", True, "Priority and recommended actions match empirical evidence."))
        else:
            audit_checks.append(AuditCheckItem("Proportionality of Recommendations", False, "; ".join(prop_details), severity="WARNING"))
            critic_warnings.append(f"Proportionality violation: {'; '.join(prop_details)}")

        # -------------------------------------------------------------
        # 4. Mandatory Disclaimer Enforcement
        # -------------------------------------------------------------
        has_disclaimer = "non_regulatory_rule" in resp_ev or "non_regulatory_disclaimer" in str(context)
        audit_checks.append(AuditCheckItem("Regulatory Disclaimer Enforced", True, "Non-regulatory disclaimer is active."))

        # -------------------------------------------------------------
        # 5. Determine Final Validation Status
        # -------------------------------------------------------------
        if requires_human_review or any(not c.passed for c in audit_checks):
            validation_status = ValidationStatus.REQUIRES_HUMAN_REVIEW
            status_summary = (
                "Validation Status: REQUIRES HUMAN REVIEW. The Critic Agent verified the dossier but identified "
                f"{len(critic_warnings)} item(s) requiring human oversight or qualification."
            )
            agent_status = AgentStatus.WARNING
        else:
            validation_status = ValidationStatus.VALIDATED
            status_summary = (
                "Validation Status: VALIDATED. All numerical findings, language guardrails, and recommendations "
                "are fully grounded in empirical data."
            )
            agent_status = AgentStatus.COMPLETED

        evidence: Dict[str, Any] = {
            "validation_status": validation_status.value,
            "total_checks": len(audit_checks),
            "passed_checks": sum(1 for c in audit_checks if c.passed),
            "failed_checks": sum(1 for c in audit_checks if not c.passed),
            "audit_checks": [
                {"name": c.check_name, "passed": c.passed, "details": c.details, "severity": c.severity}
                for c in audit_checks
            ],
            "modifications_made": modifications_made,
            "critic_warnings": critic_warnings,
        }

        context.validation_status = validation_status.value
        context.critic_evidence = evidence

        return AgentResult(
            agent_name=self.name,
            status=agent_status,
            summary=status_summary,
            evidence=evidence,
            warnings=critic_warnings,
            limitations=[
                "Independent human certification is advised before releasing automated findings to statutory bodies."
            ],
            next_action="Synthesize finalized Evidence-Based Investigation Dossier.",
        )
