"""
Multi-Station Context Agent for BreatheOps AI.

Examines cross-station readings and contextual meteorological evidence
during the event time window, strictly separating empirical data from causal hypotheses.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from breatheops.agents.base import AgentResult, AgentStatus, BaseAgent
from breatheops.analysis import POLLUTANT_UNITS


class MultiStationContextAgent(BaseAgent):
    """Multi-Station Context Agent evaluating concurrent spatial and weather conditions."""

    def __init__(self):
        super().__init__(
            name="Multi-Station Context Agent",
            description="Compares cross-station pollutant levels and examines contextual weather variables.",
        )

    def execute(self, context: Any) -> AgentResult:
        """Run multi-station comparison and weather contextualization."""
        cleaned_df: Optional[pd.DataFrame] = getattr(context, "cleaned_df", None)
        target_station: str = getattr(context, "target_station", "All")
        target_pollutant: str = getattr(context, "target_pollutant", "PM2.5")
        spike_evidence: Dict[str, Any] = getattr(context, "spike_evidence", {})
        unit = POLLUTANT_UNITS.get(target_pollutant, "µg/m³")

        warnings: List[str] = []
        limitations: List[str] = []

        if cleaned_df is None or cleaned_df.empty:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SKIPPED,
                summary="Multi-station contextual analysis skipped: No observations available.",
                warnings=["Empty dataset."],
                limitations=["No data available for spatial comparison."],
                next_action="Proceed to Response Planning.",
                evidence={"status": "NO_DATA"},
            )

        # 1. Identify stations in dataset
        if "station" in cleaned_df.columns:
            all_stations = sorted([str(s) for s in cleaned_df["station"].dropna().unique() if str(s).strip()])
        else:
            all_stations = ["Default_Station"]

        other_stations = [s for s in all_stations if s != target_station]

        evidence: Dict[str, Any] = {
            "all_stations": all_stations,
            "station_count": len(all_stations),
            "target_station": target_station,
            "other_stations": other_stations,
            "cross_station_available": len(other_stations) > 0,
            "weather_available": False,
        }

        # Handle single-station limitation
        if len(other_stations) == 0:
            limitations.append(
                f"Cross-station comparison is not possible: Dataset contains only one monitoring station ('{target_station}'). "
                "Spatial dispersion and cross-station validation require multiple concurrent stations."
            )
            summary_msg = (
                f"Cross-station comparison restricted: Only single station '{target_station}' exists in dataset. "
            )

            # Still evaluate weather context if available
            weather_summary, weather_ev = self._analyze_weather(cleaned_df, target_station, spike_evidence)
            evidence["weather_summary"] = weather_summary
            if weather_ev:
                evidence["weather_context"] = weather_ev
                evidence["weather_available"] = True
                summary_msg += weather_summary

            context.multi_station_evidence = evidence

            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.WARNING,
                summary=summary_msg,
                evidence=evidence,
                warnings=warnings,
                limitations=limitations,
                next_action="Proceed to Response Planning with single-station constraint noted.",
            )

        # 2. Multi-station comparison during relevant timeframe
        anomalies_detected = spike_evidence.get("anomalies_detected", False)
        earliest_ts = spike_evidence.get("earliest_anomaly_timestamp")
        latest_ts = spike_evidence.get("latest_anomaly_timestamp")

        # Determine evaluation window
        if anomalies_detected and earliest_ts and latest_ts and earliest_ts != "N/A":
            try:
                start_window = pd.to_datetime(earliest_ts) - pd.Timedelta(hours=2)
                end_window = pd.to_datetime(latest_ts) + pd.Timedelta(hours=2)
                window_df = cleaned_df[
                    (cleaned_df["timestamp"] >= start_window) & (cleaned_df["timestamp"] <= end_window)
                ].copy()
                window_label = f"event window ({earliest_ts} to {latest_ts})"
            except Exception:
                window_df = cleaned_df.copy()
                window_label = "entire observation period"
        else:
            window_df = cleaned_df.copy()
            window_label = "entire observation period"

        # Compare target station vs other stations
        station_window_stats: Dict[str, Dict[str, float]] = {}
        for stn in all_stations:
            sub = window_df[window_df["station"] == stn]
            if not sub.empty and target_pollutant in sub.columns:
                vals = pd.to_numeric(sub[target_pollutant], errors="coerce").dropna()
                if not vals.empty:
                    station_window_stats[stn] = {
                        "mean": round(float(vals.mean()), 2),
                        "max": round(float(vals.max()), 2),
                        "count": int(len(vals)),
                    }

        target_stats = station_window_stats.get(target_station, {"mean": 0.0, "max": 0.0, "count": 0})
        other_means = [
            stats["mean"] for stn, stats in station_window_stats.items() if stn != target_station and stats["count"] > 0
        ]
        other_maxes = [
            stats["max"] for stn, stats in station_window_stats.items() if stn != target_station and stats["count"] > 0
        ]

        if other_means:
            avg_others_mean = round(float(np.mean(other_means)), 2)
            max_others_peak = round(float(np.max(other_maxes)), 2)
            ratio_to_others = (
                round(target_stats["mean"] / avg_others_mean, 2) if avg_others_mean > 0 else 1.0
            )
        else:
            avg_others_mean = 0.0
            max_others_peak = 0.0
            ratio_to_others = 1.0

        evidence["window_evaluated"] = window_label
        evidence["station_window_stats"] = station_window_stats
        evidence["target_window_mean"] = target_stats["mean"]
        evidence["target_window_max"] = target_stats["max"]
        evidence["other_stations_window_mean"] = avg_others_mean
        evidence["other_stations_window_peak"] = max_others_peak
        evidence["ratio_target_to_others"] = ratio_to_others

        # Classify spatial distribution
        if anomalies_detected:
            if ratio_to_others >= 1.6 and target_stats["max"] > (max_others_peak * 1.3):
                spatial_classification = "ISOLATED_LOCAL_ELEVATION"
                spatial_desc = (
                    f"Elevated {target_pollutant} levels appear primarily isolated to station '{target_station}' "
                    f"(Target peak: {target_stats['max']:.1f} {unit} vs other stations' peak: {max_others_peak:.1f} {unit}, "
                    f"a {ratio_to_others:.1f}× mean ratio). Other stations in the dataset did not record comparable spikes."
                )
            elif ratio_to_others < 1.3 or max_others_peak > (target_stats["max"] * 0.75):
                spatial_classification = "CO_OCCURRING_REGIONAL_ELEVATION"
                spatial_desc = (
                    f"Co-occurring elevation was observed across multiple monitoring stations during the {window_label}. "
                    f"Other stations also recorded elevated readings (Other stations' peak: {max_others_peak:.1f} {unit}, "
                    f"mean: {avg_others_mean:.1f} {unit}), suggesting potential regional atmospheric influences."
                )
            else:
                spatial_classification = "MODERATE_LOCAL_DISPERSION"
                spatial_desc = (
                    f"Target station '{target_station}' recorded elevated readings ({target_stats['max']:.1f} {unit}), "
                    f"with moderate background elevation at other stations (Mean: {avg_others_mean:.1f} {unit})."
                )
        else:
            spatial_classification = "UNIFORM_BASELINE_OBSERVATIONS"
            spatial_desc = (
                f"Readings across all {len(all_stations)} stations remained consistent with historical baselines "
                f"during the {window_label} (Target mean: {target_stats['mean']:.1f} {unit}, Others mean: {avg_others_mean:.1f} {unit})."
            )

        evidence["spatial_classification"] = spatial_classification
        evidence["spatial_description"] = spatial_desc

        # 3. Weather variables examination
        weather_summary, weather_ev = self._analyze_weather(cleaned_df, target_station, spike_evidence)
        evidence["weather_summary"] = weather_summary
        if weather_ev:
            evidence["weather_context"] = weather_ev
            evidence["weather_available"] = True

        # Non-causal disclaimer mandatory
        limitations.append(
            "Notice on Geography & Causality: Stations are evaluated as comparative observation points within "
            "the provided dataset. In the absence of validated GIS coordinate models, geographic proximity cannot "
            "be assumed. Meteorological variables provide atmospheric context only and must not be interpreted as proving causation."
        )

        full_summary = f"{spatial_desc} {weather_summary}".strip()
        context.multi_station_evidence = evidence

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.COMPLETED,
            summary=full_summary,
            evidence=evidence,
            warnings=warnings,
            limitations=limitations,
            next_action="Invoke Response Planning Agent to formulate action recommendations.",
        )

    def _analyze_weather(
        self,
        df: pd.DataFrame,
        station: str,
        spike_evidence: Dict[str, Any],
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Extract meteorological context during the relevant time window."""
        stn_df = df[df["station"] == station] if "station" in df.columns else df

        weather_cols = [c for c in ["WSPM", "wd", "TEMP", "RAIN", "PRES"] if c in stn_df.columns]
        if not weather_cols:
            return "No meteorological sensor columns were available in the dataset.", None

        w_ev: Dict[str, Any] = {}
        anomalies_detected = spike_evidence.get("anomalies_detected", False)
        earliest_ts = spike_evidence.get("earliest_anomaly_timestamp")
        latest_ts = spike_evidence.get("latest_anomaly_timestamp")

        if anomalies_detected and earliest_ts and latest_ts and earliest_ts != "N/A":
            try:
                sub = stn_df[
                    (stn_df["timestamp"] >= pd.to_datetime(earliest_ts))
                    & (stn_df["timestamp"] <= pd.to_datetime(latest_ts))
                ]
            except Exception:
                sub = stn_df
        else:
            sub = stn_df

        notes = []
        if "WSPM" in sub.columns:
            wind_speeds = pd.to_numeric(sub["WSPM"], errors="coerce").dropna()
            if not wind_speeds.empty:
                avg_wind = round(float(wind_speeds.mean()), 2)
                min_wind = round(float(wind_speeds.min()), 2)
                w_ev["avg_wind_speed_m_s"] = avg_wind
                w_ev["min_wind_speed_m_s"] = min_wind
                if avg_wind < 1.5:
                    notes.append(f"Low wind speed recorded ({avg_wind:.1f} m/s, indicating atmospheric stagnation)")
                else:
                    notes.append(f"Moderate wind speed recorded ({avg_wind:.1f} m/s)")

        if "wd" in sub.columns:
            valid_wd = sub["wd"].dropna().astype(str)
            if not valid_wd.empty:
                mode_wd = valid_wd.mode().iloc[0]
                w_ev["predominant_wind_direction"] = mode_wd
                notes.append(f"predominant wind direction '{mode_wd}'")

        if "RAIN" in sub.columns:
            rain_vals = pd.to_numeric(sub["RAIN"], errors="coerce").dropna()
            if not rain_vals.empty:
                total_rain = round(float(rain_vals.sum()), 2)
                w_ev["total_precipitation_mm"] = total_rain
                if total_rain == 0:
                    notes.append("no precipitation")
                else:
                    notes.append(f"{total_rain:.1f} mm rain recorded")

        if "TEMP" in sub.columns:
            temp_vals = pd.to_numeric(sub["TEMP"], errors="coerce").dropna()
            if not temp_vals.empty:
                avg_temp = round(float(temp_vals.mean()), 1)
                w_ev["avg_temperature_c"] = avg_temp
                notes.append(f"mean temperature {avg_temp}°C")

        summary_text = (
            f"Meteorological Context: {', '.join(notes)} during the observation window."
            if notes
            else "Meteorological sensors were present but observations were empty."
        )

        return summary_text, w_ev
