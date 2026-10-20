"""
Pollution Spike Detection Agent for BreatheOps AI.

Executes transparent statistical baseline analysis and anomaly detection
using Median and Median Absolute Deviation (MAD), returning structured empirical evidence.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from breatheops.agents.base import AgentResult, AgentStatus, BaseAgent
from breatheops.analysis import AnomalyDetectionResult, detect_anomalies_mad, POLLUTANT_UNITS


class PollutionSpikeDetectionAgent(BaseAgent):
    """Pollution Spike Detection Agent running explainable statistical detection."""

    def __init__(self):
        super().__init__(
            name="Pollution Spike Detection Agent",
            description="Computes station-specific historical median and MAD baselines and detects anomalous spikes.",
        )

    def execute(self, context: Any) -> AgentResult:
        """Execute anomaly detection on cleaned data context."""
        cleaned_df: Optional[pd.DataFrame] = getattr(context, "cleaned_df", None)
        target_station: str = getattr(context, "target_station", "All")
        target_pollutant: str = getattr(context, "target_pollutant", "PM2.5")
        k_multiplier: float = getattr(context, "k_multiplier", 3.0)
        unit = POLLUTANT_UNITS.get(target_pollutant, "µg/m³")

        if cleaned_df is None or cleaned_df.empty:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SKIPPED,
                summary="Spike detection skipped: No clean observations available.",
                warnings=["Dataset is empty or unparseable."],
                limitations=["Statistical detection could not be performed."],
                next_action="Proceed to Response Planning with unanalyzed status.",
                evidence={"anomaly_count": 0, "status": "NO_DATA"},
            )

        # Check if station exists
        station_df = cleaned_df
        if "station" in cleaned_df.columns and target_station in cleaned_df["station"].values:
            station_df = cleaned_df[cleaned_df["station"] == target_station]

        # Call existing robust Phase 1 detection
        result: AnomalyDetectionResult = detect_anomalies_mad(
            df=station_df,
            pollutant=target_pollutant,
            station=target_station,
            k_multiplier=k_multiplier,
        )

        context.spike_detection_result = result

        # Structure empirical evidence
        evidence: Dict[str, Any] = {
            "pollutant": target_pollutant,
            "unit": unit,
            "station": target_station,
            "total_observations": result.total_observations,
            "valid_observations": result.valid_observations,
            "missing_count": result.missing_count,
            "sufficient_data": result.sufficient_data,
            "baseline_median": result.baseline_median,
            "raw_mad": result.baseline_mad,
            "dispersion_metric": result.dispersion_metric,
            "dispersion_value": result.dispersion_value,
            "threshold_multiplier": result.threshold_multiplier,
            "threshold_value": result.threshold_value,
            "min_observed": result.min_reading,
            "max_observed": result.max_reading,
            "mean_observed": result.mean_reading,
            "anomaly_count": result.anomaly_count,
            "anomalies_detected": result.anomaly_count > 0,
        }

        warnings = list(result.warnings)
        limitations: List[str] = []

        if not result.sufficient_data:
            limitations.append(
                f"Statistical baseline is constrained by limited historical records ({result.valid_observations} observed)."
            )

        # If anomalies found, capture detailed timeline
        if result.anomaly_count > 0 and not result.anomalies_df.empty:
            anom_df = result.anomalies_df
            peak_val = float(anom_df[target_pollutant].max())
            peak_ratio = peak_val / result.baseline_median if (result.baseline_median and result.baseline_median > 0) else 1.0

            earliest_ts = str(anom_df["timestamp"].iloc[0]) if "timestamp" in anom_df.columns else "N/A"
            latest_ts = str(anom_df["timestamp"].iloc[-1]) if "timestamp" in anom_df.columns else "N/A"

            evidence["peak_anomaly_value"] = round(peak_val, 2)
            evidence["peak_ratio_to_median"] = round(peak_ratio, 2)
            evidence["earliest_anomaly_timestamp"] = earliest_ts
            evidence["latest_anomaly_timestamp"] = latest_ts
            evidence["anomaly_timestamps"] = [str(ts) for ts in anom_df["timestamp"].tolist()[:20]]

            summary_msg = (
                f"Detected {result.anomaly_count} anomalous {target_pollutant} observation(s) exceeding "
                f"threshold {result.threshold_value:.1f} {unit} (Baseline Median: {result.baseline_median:.1f} {unit}). "
                f"Peak reached {peak_val:.1f} {unit} ({peak_ratio:.1f}× baseline) between {earliest_ts} and {latest_ts}."
            )
            status = AgentStatus.COMPLETED
            next_action = "Invoke Multi-Station Context Agent to assess spatial dispersion and weather context."

        elif result.valid_observations == 1:
            summary_msg = (
                f"Single observation examined for {target_pollutant} ({result.min_reading} {unit}). "
                "Baseline cannot be computed from a single row."
            )
            status = AgentStatus.WARNING
            next_action = "Invoke Multi-Station Context Agent."
        elif result.valid_observations == 0:
            summary_msg = f"No valid numeric readings available for {target_pollutant}."
            status = AgentStatus.WARNING
            next_action = "Halt spatial analysis; proceed to Response Planning."
        else:
            summary_msg = (
                f"No statistical anomalies detected for {target_pollutant} at station '{target_station}'. "
                f"All {result.valid_observations} readings stayed within baseline variability (Peak: {result.max_reading:.1f} {unit} "
                f"vs Threshold: {result.threshold_value:.1f} {unit})."
            )
            limitations.append(
                "Absence of a statistical anomaly indicates readings are within historical norms; "
                "it does not constitute an official declaration that the air is clean or safe under health standards."
            )
            status = AgentStatus.COMPLETED
            next_action = "Invoke Multi-Station Context Agent to check regional background levels."

        context.spike_evidence = evidence

        return AgentResult(
            agent_name=self.name,
            status=status,
            summary=summary_msg,
            evidence=evidence,
            warnings=warnings,
            limitations=limitations,
            next_action=next_action,
        )
