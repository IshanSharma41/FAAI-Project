"""
Planner Agent for BreatheOps AI.

Formulates an explicit investigation plan based on user selection, available
dataset schema, station counts, and data readiness.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from breatheops.agents.base import AgentResult, AgentStatus, BaseAgent
from breatheops.data_loader import CANONICAL_POLLUTANTS, CANONICAL_WEATHER


@dataclass
class PlanStep:
    """A discrete planned execution step in the investigation workflow."""
    step_id: int
    agent_name: str
    action: str
    description: str
    status: AgentStatus = AgentStatus.PENDING
    rationale: str = ""


@dataclass
class InvestigationPlan:
    """The formulated investigation plan."""
    plan_id: str
    target_station: str
    target_pollutant: str
    planned_steps: List[PlanStep] = field(default_factory=list)
    available_stations: List[str] = field(default_factory=list)
    available_pollutants: List[str] = field(default_factory=list)
    available_weather: List[str] = field(default_factory=list)
    can_proceed: bool = True
    plan_notes: List[str] = field(default_factory=list)

    def summary_markdown(self) -> str:
        lines = [
            f"### Investigation Plan: `{self.plan_id}`",
            f"- **Target**: Station `{self.target_station}`, Pollutant `{self.target_pollutant}`",
            f"- **Available Stations**: {', '.join(self.available_stations) if self.available_stations else 'None'}",
            f"- **Available Pollutants**: {', '.join(self.available_pollutants) if self.available_pollutants else 'None'}",
            "\n**Planned Execution Steps:**",
        ]
        for step in self.planned_steps:
            lines.append(
                f"{step.step_id}. **{step.agent_name}** — *{step.action}* "
                f"[{step.status.value}]: {step.description}"
            )
        if self.plan_notes:
            lines.append("\n**Plan Directives & Constraints:**")
            for note in self.plan_notes:
                lines.append(f"- ℹ️ {note}")
        return "\n".join(lines)


class PlannerAgent(BaseAgent):
    """Planner Agent producing explicit, adaptive investigation plans."""

    def __init__(self):
        super().__init__(
            name="Planner Agent",
            description="Analyzes input data profile and formulates the tailored investigation workflow.",
        )

    def execute(self, context: Any) -> AgentResult:
        """Formulate the investigation plan based on context."""
        df: Optional[pd.DataFrame] = getattr(context, "raw_df", None)
        target_station: str = getattr(context, "target_station", "All")
        target_pollutant: str = getattr(context, "target_pollutant", "PM2.5")
        plan_id = f"PLAN-{abs(hash(str(target_station) + str(target_pollutant) + str(len(df) if df is not None else 0))) % 100000:05d}"

        plan = InvestigationPlan(
            plan_id=plan_id,
            target_station=target_station,
            target_pollutant=target_pollutant,
        )

        # 1. Inspect dataset structure
        if df is None or df.empty:
            plan.can_proceed = False
            plan.plan_notes.append("Dataset is empty. Execution cannot proceed beyond Data Quality check.")
            plan.planned_steps = [
                PlanStep(
                    step_id=1,
                    agent_name="Data Quality Agent",
                    action="Validate Ingestion",
                    description="Verify file integrity and document missing records.",
                    status=AgentStatus.PENDING,
                    rationale="Required step for all datasets.",
                ),
                PlanStep(
                    step_id=2,
                    agent_name="Response Planning Agent",
                    action="Formulate Incident Response",
                    description="Issue data acquisition advisory due to empty input.",
                    status=AgentStatus.PENDING,
                    rationale="Provides corrective guidance when input is unviable.",
                ),
                PlanStep(
                    step_id=3,
                    agent_name="Critic and Safety Agent",
                    action="Safety & Consistency Review",
                    description="Verify that no unsupported conclusions are rendered from missing data.",
                    status=AgentStatus.PENDING,
                    rationale="Enforces verification integrity.",
                ),
            ]
            context.plan = plan
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.WARNING,
                summary=f"Investigation plan {plan_id} formulated with restricted scope due to empty dataset.",
                evidence={"plan": plan, "can_proceed": False},
                warnings=["Empty input dataset supplied."],
                next_action="Invoke Data Quality Agent to record data ingestion failure.",
            )

        # Detect columns
        cols = [c.strip() for c in df.columns]
        avail_pollutants = [p for p in CANONICAL_POLLUTANTS if any(c.lower() == p.lower() or c.lower().replace(".", "").replace("_", "") == p.lower().replace(".", "").replace("_", "") for c in cols)]
        avail_weather = [w for w in CANONICAL_WEATHER if any(c.lower() == w.lower() for c in cols)]

        if "station" in df.columns:
            avail_stations = sorted([str(s) for s in df["station"].dropna().unique() if str(s).strip()])
        else:
            avail_stations = ["Default_Station"]

        plan.available_stations = avail_stations
        plan.available_pollutants = avail_pollutants
        plan.available_weather = avail_weather

        # 2. Build adaptive steps
        steps: List[PlanStep] = []
        step_idx = 1

        # Step 1: Data Quality Agent
        steps.append(
            PlanStep(
                step_id=step_idx,
                agent_name="Data Quality Agent",
                action="Data Integrity & Profiling",
                description="Profile missing readings, audit timestamps, and evaluate statistical sample sufficiency.",
                status=AgentStatus.PENDING,
                rationale="Validates whether data meets requirements for baseline computation.",
            )
        )
        step_idx += 1

        # Check if target pollutant exists or can be matched
        has_pollutant = any(p.lower() == target_pollutant.lower() for p in avail_pollutants) or (target_pollutant in df.columns)

        if not has_pollutant:
            plan.can_proceed = False
            plan.plan_notes.append(
                f"Target pollutant '{target_pollutant}' is not present in the dataset. "
                "Spike detection and spatial analysis steps will be omitted."
            )
        else:
            # Step 2: Pollution Spike Detection Agent
            steps.append(
                PlanStep(
                    step_id=step_idx,
                    agent_name="Pollution Spike Detection Agent",
                    action="Statistical Anomaly Detection",
                    description=f"Establish robust station baseline and detect {target_pollutant} spikes using Median and MAD.",
                    status=AgentStatus.PENDING,
                    rationale="Applies explainable, transparent statistics to detect unusual pollution levels.",
                )
            )
            step_idx += 1

            # Step 3: Multi-Station Context Agent
            can_cross_station = len(avail_stations) > 1
            if can_cross_station:
                multi_desc = f"Compare {target_pollutant} levels across {len(avail_stations)} stations and examine meteorological context."
                multi_rationale = "Determines whether the event was localized to one station or regional across multiple stations."
            else:
                multi_desc = f"Evaluate localized weather context for station '{target_station}' (cross-station comparison omitted: only 1 station)."
                multi_rationale = "Single-station dataset restricts spatial comparison; reviews available weather only."
                plan.plan_notes.append("Single-station dataset detected: cross-station comparison will be marked as restricted.")

            steps.append(
                PlanStep(
                    step_id=step_idx,
                    agent_name="Multi-Station Context Agent",
                    action="Spatial & Meteorological Contextualization",
                    description=multi_desc,
                    status=AgentStatus.PENDING,
                    rationale=multi_rationale,
                )
            )
            step_idx += 1

        # Step 4: Response Planning Agent
        steps.append(
            PlanStep(
                step_id=step_idx,
                agent_name="Response Planning Agent",
                action="Prioritization & Advisory Formulation",
                description="Synthesize findings into an investigation priority rating and recommend actionable follow-up tasks.",
                status=AgentStatus.PENDING,
                rationale="Transforms empirical findings into structured advisory next steps.",
            )
        )
        step_idx += 1

        # Step 5: Critic and Safety Agent
        steps.append(
            PlanStep(
                step_id=step_idx,
                agent_name="Critic and Safety Agent",
                action="Safety, Consistency & Proportionality Audit",
                description="Verify factual statements against analysis numbers and eliminate ungrounded causal or regulatory claims.",
                status=AgentStatus.PENDING,
                rationale="Ensures scientific integrity, prevents hallucinations, and enforces non-regulatory disclaimers.",
            )
        )

        plan.planned_steps = steps
        context.plan = plan

        summary_msg = (
            f"Formulated investigation plan {plan_id} with {len(steps)} sequential stages "
            f"for station '{target_station}' and pollutant '{target_pollutant}'."
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.COMPLETED,
            summary=summary_msg,
            evidence={
                "plan_id": plan_id,
                "target_station": target_station,
                "target_pollutant": target_pollutant,
                "step_count": len(steps),
                "available_stations_count": len(avail_stations),
                "available_pollutants": avail_pollutants,
                "available_weather": avail_weather,
            },
            next_action="Execute Data Quality Agent to audit dataset integrity.",
        )
