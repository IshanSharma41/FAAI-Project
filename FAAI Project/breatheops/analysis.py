"""
Statistical anomaly detection and baseline analysis module for BreatheOps AI.

Implements transparent, explainable statistical methods for identifying
unusually high air pollutant measurements using station-specific historical
baselines based on Median and Median Absolute Deviation (MAD).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


POLLUTANT_UNITS: Dict[str, str] = {
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "SO2": "µg/m³",
    "NO2": "µg/m³",
    "CO": "µg/m³",
    "O3": "µg/m³",
}

# Normal distribution scale factor for MAD consistency: 1 / norm.ppf(0.75) ≈ 1.4826
MAD_SCALE_FACTOR: float = 1.482602218505602
DEFAULT_K_MULTIPLIER: float = 3.0
DEFAULT_MIN_OBSERVATIONS: int = 24


@dataclass
class AnomalyDetectionResult:
    """Structured result containing statistical metrics and detected anomalies."""
    pollutant: str
    station: str
    total_observations: int = 0
    valid_observations: int = 0
    missing_count: int = 0
    missing_percentage: float = 0.0
    sufficient_data: bool = False
    min_required_observations: int = DEFAULT_MIN_OBSERVATIONS

    # Baseline statistics
    baseline_median: Optional[float] = None
    baseline_mad: Optional[float] = None
    dispersion_metric: str = "Scaled MAD"
    dispersion_value: Optional[float] = None
    threshold_multiplier: float = DEFAULT_K_MULTIPLIER
    threshold_value: Optional[float] = None

    # Observed readings summary
    min_reading: Optional[float] = None
    max_reading: Optional[float] = None
    mean_reading: Optional[float] = None

    # Anomalies
    anomaly_count: int = 0
    anomalies_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    analyzed_series_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    # Metadata & text
    warnings: List[str] = field(default_factory=list)
    plain_language_finding: str = ""
    rule_description: str = ""
    non_regulatory_disclaimer: str = (
        "Statistical anomaly detection identifies measurements that deviate significantly from "
        "station baseline norms. This analysis does not represent an official regulatory violation, "
        "statutory standard compliance determination, or official environmental emergency classification."
    )

    def summary_markdown(self) -> str:
        """Produce formatted markdown overview of the investigation finding."""
        unit = POLLUTANT_UNITS.get(self.pollutant, "")
        lines = []
        lines.append(f"### Preliminary Finding: {self.pollutant} @ {self.station}")
        lines.append(f"\n{self.plain_language_finding}\n")

        lines.append("#### Statistical Baseline & Parameters")
        lines.append(f"- **Observations Examined**: {self.total_observations} (Valid: {self.valid_observations}, Missing: {self.missing_count})")
        lines.append(f"- **Data Sufficiency**: {'Sufficient' if self.sufficient_data else '⚠️ Insufficient'}")
        
        if self.baseline_median is not None:
            lines.append(f"- **Historical Baseline (Median)**: {self.baseline_median:.2f} {unit}")
            lines.append(f"- **Dispersion Metric**: {self.dispersion_metric} = {self.dispersion_value:.2f} {unit}")
            lines.append(f"- **Detection Rule**: {self.rule_description}")
            if self.threshold_value is not None:
                lines.append(f"- **Anomaly Threshold**: > {self.threshold_value:.2f} {unit}")

        lines.append(f"- **Anomalies Detected**: {self.anomaly_count}")

        if self.warnings:
            lines.append("\n**Investigation Warnings & Limitations:**")
            for w in self.warnings:
                lines.append(f"- ⚠️ {w}")

        lines.append(f"\n> ℹ️ *Note: {self.non_regulatory_disclaimer}*")
        return "\n".join(lines)


def calculate_robust_baseline(
    values: np.ndarray,
) -> Tuple[float, float, str, float]:
    """Compute robust baseline (median) and scale estimator with fallback.

    Returns:
        Tuple of (median, mad, dispersion_metric_name, effective_scale).
    """
    median = float(np.median(values))
    abs_deviations = np.abs(values - median)
    raw_mad = float(np.median(abs_deviations))
    scaled_mad = raw_mad * MAD_SCALE_FACTOR

    # Robust scale fallback if MAD is zero (e.g. repetitive low values or stepped sensors)
    if scaled_mad > 1e-5:
        return median, raw_mad, "Scaled MAD (1.4826 × MAD)", scaled_mad

    # Fallback 1: Mean Absolute Deviation from Median
    mean_ad = float(np.mean(abs_deviations))
    if mean_ad > 1e-5:
        scale_mean_ad = mean_ad * 1.2533  # asymptotic factor for normal distribution
        return median, raw_mad, "Mean Absolute Deviation Fallback", scale_mean_ad

    # Fallback 2: Standard Deviation
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if std_val > 1e-5:
        return median, raw_mad, "Sample Std. Dev. Fallback", std_val

    # Fallback 3: Constant series
    return median, raw_mad, "Zero Dispersion (Constant Values)", 0.0


def detect_anomalies_mad(
    df: pd.DataFrame,
    pollutant: str,
    station: Optional[str] = None,
    k_multiplier: float = DEFAULT_K_MULTIPLIER,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> AnomalyDetectionResult:
    """Perform explainable statistical anomaly detection on air quality observations.

    Parameters:
        df: Input DataFrame containing air-quality observations.
        pollutant: Name of the pollutant column (e.g., 'PM2.5', 'PM10', 'NO2').
        station: Optional station identifier. If None, the first available station is used.
        k_multiplier: Multiplier for the dispersion threshold (default 3.0).
        min_observations: Minimum valid data points needed for a reliable baseline.

    Returns:
        AnomalyDetectionResult with statistical parameters and detected anomalies.
    """
    result = AnomalyDetectionResult(
        pollutant=pollutant,
        station=station or "All",
        threshold_multiplier=k_multiplier,
        min_required_observations=min_observations,
    )

    unit = POLLUTANT_UNITS.get(pollutant, "µg/m³")

    # 1. Basic validation of dataframe
    if df is None or df.empty:
        result.warnings.append("Provided dataset is empty.")
        result.plain_language_finding = (
            "Investigation could not run because the provided dataset contains no rows."
        )
        return result

    if pollutant not in df.columns:
        result.warnings.append(f"Pollutant column '{pollutant}' not found in dataset.")
        result.plain_language_finding = (
            f"Investigation halted: Pollutant '{pollutant}' was not found in the input data."
        )
        return result

    # 2. Filter by station if station column exists
    working_df = df.copy()
    if "station" in working_df.columns:
        avail_stations = working_df["station"].dropna().unique().tolist()
        if station and station in avail_stations:
            working_df = working_df[working_df["station"] == station].copy()
            result.station = str(station)
        elif len(avail_stations) > 0:
            chosen = avail_stations[0]
            working_df = working_df[working_df["station"] == chosen].copy()
            result.station = str(chosen)
            if len(avail_stations) > 1:
                result.warnings.append(
                    f"Multiple stations present. Defaulting analysis to station '{chosen}'."
                )
    else:
        result.station = "Default_Station"

    total_rows = len(working_df)
    result.total_observations = total_rows

    # Check for single-row dataset
    if total_rows == 1:
        result.sufficient_data = False
        val = working_df[pollutant].iloc[0]
        result.warnings.append(
            "Dataset contains only 1 observation. Historical trend and cross-station analysis "
            "are not possible from this file alone. Insufficient historical evidence exists to establish a baseline."
        )
        result.plain_language_finding = (
            f"Preliminary Investigation Finding: Insufficient historical evidence. The file contains only a single "
            f"measurement for station '{result.station}' ({pollutant} = {val} {unit}). A valid statistical baseline "
            "requires recurring observations over time to separate ordinary variation from abnormal events."
        )
        if pd.notna(val):
            result.valid_observations = 1
            result.min_reading = float(val)
            result.max_reading = float(val)
            result.mean_reading = float(val)
        return result

    # 3. Clean and isolate pollutant series
    working_df[pollutant] = pd.to_numeric(working_df[pollutant], errors="coerce")
    valid_mask = working_df[pollutant].notna()
    valid_df = working_df[valid_mask].copy()

    valid_count = len(valid_df)
    missing_count = total_rows - valid_count
    result.valid_observations = valid_count
    result.missing_count = missing_count
    result.missing_percentage = round((missing_count / total_rows) * 100.0, 2) if total_rows > 0 else 0.0

    if result.missing_percentage > 20.0:
        result.warnings.append(
            f"High missing data rate: {result.missing_percentage:.1f}% of {pollutant} observations are null. "
            "Statistical baseline may have reduced precision."
        )

    if valid_count == 0:
        result.sufficient_data = False
        result.warnings.append(f"No valid numeric readings found for '{pollutant}' at station '{result.station}'.")
        result.plain_language_finding = (
            f"Preliminary Investigation Finding: All observations for '{pollutant}' at station '{result.station}' "
            "are missing or non-numeric. No baseline or anomaly detection can be performed."
        )
        return result

    # 4. Check sufficiency
    numeric_vals = valid_df[pollutant].values.astype(float)
    result.min_reading = round(float(np.min(numeric_vals)), 2)
    result.max_reading = round(float(np.max(numeric_vals)), 2)
    result.mean_reading = round(float(np.mean(numeric_vals)), 2)

    if valid_count < min_observations:
        result.sufficient_data = False
        result.warnings.append(
            f"Only {valid_count} valid observations available for station '{result.station}'. "
            f"A minimum of {min_observations} hourly readings is recommended for robust baseline estimation."
        )
    else:
        result.sufficient_data = True

    # 5. Compute robust baseline and dispersion
    median, raw_mad, dispersion_name, effective_scale = calculate_robust_baseline(numeric_vals)
    result.baseline_median = round(median, 2)
    result.baseline_mad = round(raw_mad, 2)
    result.dispersion_metric = dispersion_name
    result.dispersion_value = round(effective_scale, 2)

    result.rule_description = (
        f"Median + {k_multiplier:.1f} × ({dispersion_name})"
    )

    if effective_scale <= 1e-5:
        # Perfectly flat readings
        threshold = median + 1e-3
        result.warnings.append(
            "Pollutant readings have zero dispersion (constant values). Baseline threshold rule adjusted."
        )
    else:
        threshold = median + (k_multiplier * effective_scale)

    result.threshold_value = round(threshold, 2)

    # 6. Flag anomalies
    # Calculate robust z-score: (x - median) / effective_scale
    if effective_scale > 1e-5:
        working_df["robust_z_score"] = (working_df[pollutant] - median) / effective_scale
    else:
        working_df["robust_z_score"] = 0.0

    working_df["baseline_median"] = median
    working_df["threshold"] = threshold
    working_df["is_anomaly"] = (working_df[pollutant] > threshold) & (working_df[pollutant].notna())

    # Build clean anomaly subset
    anom_rows = working_df[working_df["is_anomaly"]].copy()
    result.anomaly_count = len(anom_rows)

    if result.anomaly_count > 0:
        anom_rows["difference_from_baseline"] = anom_rows[pollutant] - median
        anom_rows["ratio_to_median"] = anom_rows[pollutant] / np.where(median > 0, median, 1.0)

        # Select relevant columns for reporting
        cols_to_keep = []
        for col in ["timestamp", "station", pollutant, "baseline_median", "threshold",
                    "difference_from_baseline", "ratio_to_median", "robust_z_score",
                    "TEMP", "WSPM", "RAIN", "wd"]:
            if col in anom_rows.columns:
                cols_to_keep.append(col)

        result.anomalies_df = anom_rows[cols_to_keep].copy()

    result.analyzed_series_df = working_df

    # 7. Generate plain-language finding
    if not result.sufficient_data:
        if result.anomaly_count == 0:
            finding_text = (
                f"Preliminary Investigation Finding: Limited observation sample ({valid_count} valid readings vs. "
                f"{min_observations} recommended). No statistical anomalies were identified at or above the preliminary "
                f"threshold of {threshold:.1f} {unit}. However, this result should be treated as provisional due to "
                "insufficient historical sample size."
            )
        else:
            finding_text = (
                f"Preliminary Investigation Finding: Caution — {result.anomaly_count} high reading(s) detected, "
                f"but based on an undersized historical baseline ({valid_count} readings). Baseline reliability is limited. "
                "Additional continuous monitoring is required before confirming a sustained pollution anomaly."
            )
    else:
        if result.anomaly_count == 0:
            finding_text = (
                f"Preliminary Investigation Finding: No statistical anomalies were detected for {pollutant} at station "
                f"'{result.station}' using the robust baseline rule [{result.rule_description}]. "
                f"Across all {valid_count} analyzed hourly observations, pollutant levels remained within standard "
                f"variability (observed peak: {result.max_reading:.1f} {unit} vs. baseline median: {median:.1f} {unit}, "
                f"threshold: {threshold:.1f} {unit}). "
                "Notice: Absence of a statistical anomaly indicates normal station variation; it is not a declaration "
                "that the air is clean or safe under public health regulations."
            )
        else:
            peak_anom = float(anom_rows[pollutant].max())
            peak_ratio = peak_anom / median if median > 0 else 0.0
            earliest = anom_rows["timestamp"].iloc[0] if "timestamp" in anom_rows.columns else "N/A"
            latest = anom_rows["timestamp"].iloc[-1] if "timestamp" in anom_rows.columns else "N/A"

            time_range_str = f"between {earliest} and {latest}" if earliest != latest else f"at {earliest}"

            finding_text = (
                f"Preliminary Investigation Finding: {result.anomaly_count} unusual pollution measurement(s) detected "
                f"for {pollutant} at station '{result.station}'. The observed peak reached {peak_anom:.1f} {unit} "
                f"({peak_ratio:.1f}× station baseline median of {median:.1f} {unit}), exceeding the statistical "
                f"anomaly threshold of {threshold:.1f} {unit}. "
                f"Elevated readings occurred {time_range_str}. "
                "Weather correlation (e.g. atmospheric stagnation) and cross-station confirmation should be reviewed "
                "in Phase 2 to determine if this event represents localized source emissions or regional transport. "
                "Notice: A statistical anomaly does not constitute an official regulatory violation or statutory emergency."
            )

    result.plain_language_finding = finding_text
    return result
