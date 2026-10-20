"""
Data Quality Agent for BreatheOps AI.

Audits dataset integrity, missingness, timestamp validity, and temporal
sample sufficiency, determining whether downstream investigation can proceed.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from breatheops.agents.base import AgentResult, AgentStatus, BaseAgent
from breatheops.data_loader import DataValidationReport, load_air_quality_csv


class DataQualityAgent(BaseAgent):
    """Data Quality Agent validating records and assessing statistical viability."""

    def __init__(self):
        super().__init__(
            name="Data Quality Agent",
            description="Performs multi-dimensional data integrity, missingness, and sufficiency audits.",
        )

    def execute(self, context: Any) -> AgentResult:
        """Run data quality assessment on the current context."""
        raw_df: Optional[pd.DataFrame] = getattr(context, "raw_df", None)
        target_station: str = getattr(context, "target_station", "All")
        target_pollutant: str = getattr(context, "target_pollutant", "PM2.5")
        is_synthetic: bool = getattr(context, "is_synthetic", False)

        warnings: List[str] = []
        limitations: List[str] = []

        # 1. Check if DataFrame is already loaded or needs validation
        report: Optional[DataValidationReport] = getattr(context, "validation_report", None)
        cleaned_df = getattr(context, "cleaned_df", None)

        if cleaned_df is None or report is None:
            if raw_df is None or raw_df.empty:
                evidence = {"decision": "BLOCKED", "total_rows": 0, "usable_rows": 0}
                context.data_quality_decision = "BLOCKED"
                context.data_quality_evidence = evidence
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.FAILED,
                    summary="Data validation blocked: Dataset contains 0 rows or is empty.",
                    warnings=["Empty dataset cannot be analyzed."],
                    limitations=["No data available for temporal or statistical processing."],
                    next_action="Halt investigation and request a valid dataset.",
                    evidence=evidence,
                )

            # Validate using existing Phase 1 loader
            import io
            csv_buf = io.StringIO()
            raw_df.to_csv(csv_buf, index=False)
            csv_buf.seek(0)
            cleaned_df, report = load_air_quality_csv(csv_buf, is_synthetic=is_synthetic)
            context.cleaned_df = cleaned_df
            context.validation_report = report

        if not report.is_valid or cleaned_df is None or cleaned_df.empty:
            err_msg = "; ".join(report.errors) if report.errors else "Dataset validation failed."
            evidence = {
                "decision": "BLOCKED",
                "total_rows": report.total_rows,
                "usable_rows": report.usable_rows,
                "errors": report.errors,
            }
            context.data_quality_decision = "BLOCKED"
            context.data_quality_evidence = evidence
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                summary=f"Data Quality Gate Failed: {err_msg}",
                warnings=report.warnings,
                limitations=["Investigation halted due to unrecoverable data errors."],
                next_action="Stop investigation. User must upload a file with required temporal and pollutant fields.",
                evidence=evidence,
            )

        warnings.extend(report.warnings)

        # 2. Check station-specific observations for target station
        if target_station in cleaned_df["station"].values:
            station_df = cleaned_df[cleaned_df["station"] == target_station]
        else:
            station_df = cleaned_df

        station_rows = len(station_df)

        # Check target pollutant presence and missingness
        if target_pollutant not in station_df.columns:
            limitations.append(f"Target pollutant '{target_pollutant}' is missing from station observations.")
            evidence = {"decision": "BLOCKED", "missing_required_column": target_pollutant}
            context.data_quality_decision = "BLOCKED"
            context.data_quality_evidence = evidence
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                summary=f"Data Quality Audit: Target pollutant '{target_pollutant}' not found in dataset.",
                warnings=[f"Missing column '{target_pollutant}'."],
                limitations=limitations,
                next_action="Halt spike detection; initiate response planning for data recovery.",
                evidence=evidence,
            )

        valid_readings_series = pd.to_numeric(station_df[target_pollutant], errors="coerce").dropna()
        valid_count = len(valid_readings_series)
        missing_count = station_rows - valid_count
        missing_pct = round((missing_count / station_rows) * 100.0, 2) if station_rows > 0 else 100.0

        # Evaluate sufficiency and decision
        decision = "CAN_PROCEED"
        status = AgentStatus.COMPLETED

        if report.is_single_row or station_rows == 1:
            decision = "PROCEED_WITH_LIMITATIONS"
            status = AgentStatus.WARNING
            limitations.append(
                "Dataset contains only 1 observation. Historical trend and baseline establishment "
                "are not statistically possible from this single row alone."
            )
        elif valid_count < 24:
            decision = "PROCEED_WITH_LIMITATIONS"
            status = AgentStatus.WARNING
            limitations.append(
                f"Small sample size: Only {valid_count} valid hourly observations for '{target_pollutant}'. "
                "At least 24 hours of data are recommended for reliable diurnal baseline estimation."
            )

        if missing_pct > 25.0:
            decision = "PROCEED_WITH_LIMITATIONS"
            status = AgentStatus.WARNING
            warnings.append(
                f"High missingness for {target_pollutant}: {missing_pct:.1f}% missing ({missing_count} rows)."
            )
            limitations.append(f"Pollutant {target_pollutant} baseline accuracy is impaired by high missing data rate.")

        if report.invalid_timestamp_count > 0:
            warnings.append(f"Excluded {report.invalid_timestamp_count} record(s) with invalid timestamps.")

        if report.duplicate_rows > 0:
            warnings.append(f"Deduplicated {report.duplicate_rows} duplicate record(s).")

        evidence = {
            "decision": decision,
            "total_rows": report.total_rows,
            "usable_rows": report.usable_rows,
            "station_rows": station_rows,
            "valid_target_readings": valid_count,
            "missing_target_readings": missing_count,
            "missing_target_percentage": missing_pct,
            "invalid_timestamps": report.invalid_timestamp_count,
            "duplicate_records": report.duplicate_rows,
            "available_stations": report.available_stations,
            "available_pollutants": report.available_pollutants,
            "is_single_row": report.is_single_row,
            "is_synthetic": report.is_synthetic,
        }

        context.data_quality_decision = decision
        context.data_quality_evidence = evidence

        summary_msg = (
            f"Audited {report.usable_rows:,} usable observations across {len(report.available_stations)} station(s). "
            f"Target '{target_pollutant}' has {valid_count} valid readings ({missing_pct:.1f}% missing). "
            f"Audit Decision: {decision.replace('_', ' ')}."
        )

        return AgentResult(
            agent_name=self.name,
            status=status,
            summary=summary_msg,
            evidence=evidence,
            warnings=warnings,
            limitations=limitations,
            next_action=(
                "Proceed to Pollution Spike Detection Agent."
                if decision != "BLOCKED"
                else "Halt analytical workflow and trigger Data Audit Response."
            ),
        )
