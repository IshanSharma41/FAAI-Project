"""
Data loading and validation module for BreatheOps AI.

Provides safe reading of air-quality CSV files, schema normalization,
timestamp reconstruction, missing value profiling, and data-integrity validation.
"""

from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Canonical column names
CANONICAL_POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
CANONICAL_WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"]
DATE_COMPONENTS = ["year", "month", "day", "hour"]

# Alias dictionary: mapped from simplified lowercase string to canonical name
COLUMN_ALIASES: Dict[str, str] = {
    # Pollutants
    "pm2.5": "PM2.5",
    "pm25": "PM2.5",
    "pm2_5": "PM2.5",
    "pm_25": "PM2.5",
    "pm2.5(ug/m3)": "PM2.5",
    "pm25(ug/m3)": "PM2.5",
    "pm10": "PM10",
    "pm_10": "PM10",
    "pm10(ug/m3)": "PM10",
    "so2": "SO2",
    "so_2": "SO2",
    "so2(ug/m3)": "SO2",
    "no2": "NO2",
    "no_2": "NO2",
    "no2(ug/m3)": "NO2",
    "co": "CO",
    "co(mg/m3)": "CO",
    "co(ug/m3)": "CO",
    "o3": "O3",
    "ozone": "O3",
    "o3(ug/m3)": "O3",
    # Weather
    "temp": "TEMP",
    "temperature": "TEMP",
    "pres": "PRES",
    "pressure": "PRES",
    "dewp": "DEWP",
    "dewpoint": "DEWP",
    "dew_point": "DEWP",
    "rain": "RAIN",
    "precipitation": "RAIN",
    "prcp": "RAIN",
    "wd": "wd",
    "wind_direction": "wd",
    "wind_dir": "wd",
    "wspm": "WSPM",
    "wind_speed": "WSPM",
    "windspeed": "WSPM",
    "wspd": "WSPM",
    # Metadata & Date components
    "station": "station",
    "station_name": "station",
    "site": "station",
    "location": "station",
    "year": "year",
    "yr": "year",
    "month": "month",
    "mon": "month",
    "mo": "month",
    "day": "day",
    "d": "day",
    "hour": "hour",
    "hr": "hour",
    "h": "hour",
    "datetime": "timestamp",
    "timestamp": "timestamp",
    "date_time": "timestamp",
    "date": "timestamp",
    "time": "timestamp",
}


@dataclass
class DataValidationReport:
    """Summary of data loading and validation results."""
    is_valid: bool = False
    total_rows: int = 0
    usable_rows: int = 0
    invalid_timestamp_count: int = 0
    duplicate_rows: int = 0
    missing_pollutant_counts: Dict[str, int] = field(default_factory=dict)
    missing_pollutant_percentages: Dict[str, float] = field(default_factory=dict)
    available_stations: List[str] = field(default_factory=list)
    available_pollutants: List[str] = field(default_factory=list)
    available_weather: List[str] = field(default_factory=list)
    missing_required_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    is_single_row: bool = False
    is_synthetic: bool = False

    def summary_markdown(self) -> str:
        """Generate human-readable summary of the validation report."""
        lines = []
        status_symbol = "✅" if self.is_valid else "❌"
        lines.append(f"### {status_symbol} Data Validation Summary")
        lines.append(f"- **Total rows loaded**: {self.total_rows}")
        lines.append(f"- **Usable observations (valid timestamp)**: {self.usable_rows}")
        if self.invalid_timestamp_count > 0:
            lines.append(f"- **Invalid / unparseable timestamps**: {self.invalid_timestamp_count}")
        if self.duplicate_rows > 0:
            lines.append(f"- **Duplicate observations**: {self.duplicate_rows}")
        lines.append(f"- **Stations detected**: {', '.join(self.available_stations) if self.available_stations else 'None'}")
        lines.append(f"- **Pollutants available**: {', '.join(self.available_pollutants) if self.available_pollutants else 'None'}")
        
        if self.missing_pollutant_percentages:
            missing_info = [f"{k}: {pct:.1f}% ({self.missing_pollutant_counts[k]} rows)"
                            for k, pct in self.missing_pollutant_percentages.items() if pct > 0]
            if missing_info:
                lines.append(f"- **Missing pollutant readings**: {', '.join(missing_info)}")

        if self.warnings:
            lines.append("\n**Warnings & Limitations:**")
            for w in self.warnings:
                lines.append(f"- ⚠️ {w}")

        if self.errors:
            lines.append("\n**Errors Encountered:**")
            for e in self.errors:
                lines.append(f"- 🛑 {e}")

        return "\n".join(lines)


def normalize_column_name(col: str) -> str:
    """Normalize a column name by stripping whitespace, converting case, and matching aliases."""
    cleaned = str(col).strip()
    simplified = cleaned.lower().replace(" ", "").replace("_", "").replace("-", "")

    # Direct check on simplified
    if simplified in COLUMN_ALIASES:
        return COLUMN_ALIASES[simplified]

    # Check against known aliases directly
    lowered = cleaned.lower()
    if lowered in COLUMN_ALIASES:
        return COLUMN_ALIASES[lowered]

    # Return trimmed original name if no standard alias matched
    return cleaned


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe column names to canonical forms, preserving unknown columns."""
    df_copy = df.copy()
    new_cols = [normalize_column_name(c) for c in df_copy.columns]
    df_copy.columns = new_cols
    return df_copy


def load_air_quality_csv(
    source: Union[str, Path, BytesIO, StringIO, Any],
    is_synthetic: bool = False,
) -> Tuple[Optional[pd.DataFrame], DataValidationReport]:
    """Safely load and validate an air-quality CSV file or stream.

    Parameters:
        source: File path, file buffer, StringIO, or uploaded Streamlit file.
        is_synthetic: Whether the data source is flagged as synthetic demonstration data.

    Returns:
        Tuple of (Cleaned DataFrame or None, DataValidationReport).
    """
    report = DataValidationReport(is_synthetic=is_synthetic)

    # 1. Read the CSV safely
    try:
        if isinstance(source, (str, Path)):
            path_obj = Path(source)
            if not path_obj.exists():
                report.errors.append(f"File not found: {source}")
                return None, report
            df = pd.read_csv(source)
        else:
            # Buffer or uploaded file
            try:
                df = pd.read_csv(source)
            except UnicodeDecodeError:
                if hasattr(source, "seek"):
                    source.seek(0)
                df = pd.read_csv(source, encoding="latin1")

    except pd.errors.EmptyDataError:
        report.errors.append("Uploaded file is empty.")
        return None, report
    except Exception as exc:
        report.errors.append(f"Failed to read CSV: {str(exc)}")
        return None, report

    if df.empty:
        report.errors.append("Dataset contains 0 rows.")
        return None, report

    report.total_rows = len(df)

    # Check if single row
    if report.total_rows == 1:
        report.is_single_row = True
        report.warnings.append(
            "Dataset contains only 1 observation. Historical trend, time-series baseline "
            "establishment, and cross-station analysis are not possible from a single row."
        )

    # 2. Normalize columns
    df = normalize_dataframe_columns(df)

    # 3. Check for stations
    if "station" not in df.columns:
        df["station"] = "Default_Station"
        report.warnings.append(
            "Column 'station' was missing. Assigned default station name 'Default_Station'."
        )
    else:
        df["station"] = df["station"].astype(str).str.strip()

    report.available_stations = sorted([s for s in df["station"].dropna().unique() if s != ""])
    if not report.available_stations:
        report.available_stations = ["Default_Station"]
        df["station"] = "Default_Station"

    # 4. Construct / parse timestamp
    has_timestamp = "timestamp" in df.columns
    has_date_parts = all(c in df.columns for c in DATE_COMPONENTS)

    if not has_timestamp and not has_date_parts:
        missing_parts = [c for c in DATE_COMPONENTS if c not in df.columns]
        report.missing_required_columns.extend(missing_parts)
        report.errors.append(
            "Missing temporal reference. Must supply either a 'timestamp' column or "
            f"date component columns: {DATE_COMPONENTS}. (Missing: {missing_parts})"
        )
        return None, report

    if not has_timestamp and has_date_parts:
        # Build timestamp from year, month, day, hour
        try:
            years = pd.to_numeric(df["year"], errors="coerce")
            months = pd.to_numeric(df["month"], errors="coerce")
            days = pd.to_numeric(df["day"], errors="coerce")
            hours = pd.to_numeric(df["hour"], errors="coerce")

            df["timestamp"] = pd.to_datetime(
                {
                    "year": years,
                    "month": months,
                    "day": days,
                    "hour": hours,
                },
                errors="coerce",
            )
        except Exception as exc:
            report.errors.append(f"Failed to construct timestamps from date components: {exc}")
            return None, report
    else:
        # Timestamp column exists, parse it safely
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Assess timestamp validity
    invalid_ts = df["timestamp"].isna()
    report.invalid_timestamp_count = int(invalid_ts.sum())
    if report.invalid_timestamp_count > 0:
        report.warnings.append(
            f"Found {report.invalid_timestamp_count} row(s) with invalid or unparseable timestamps. "
            "These rows cannot be included in temporal time-series analysis."
        )

    # Usable rows are those with valid timestamp
    usable_df = df[~invalid_ts].copy()
    report.usable_rows = len(usable_df)

    if report.usable_rows == 0:
        report.errors.append("No rows with valid timestamps could be recovered.")
        return None, report

    # 5. Check duplicate observations
    duplicate_mask = usable_df.duplicated(subset=["station", "timestamp"], keep=False)
    report.duplicate_rows = int(duplicate_mask.sum())
    if report.duplicate_rows > 0:
        report.warnings.append(
            f"Found {report.duplicate_rows} duplicate observation(s) with matching station and timestamp. "
            "Preserving first occurrences for analysis."
        )
        # Drop duplicates for analysis while keeping first
        usable_df = usable_df.drop_duplicates(subset=["station", "timestamp"], keep="first")

    # 6. Profile pollutants
    found_pollutants = [p for p in CANONICAL_POLLUTANTS if p in usable_df.columns]
    report.available_pollutants = found_pollutants

    if not found_pollutants:
        report.missing_required_columns.extend(CANONICAL_POLLUTANTS)
        report.errors.append(
            "No recognized pollutant columns found. Expected at least one of: "
            f"{', '.join(CANONICAL_POLLUTANTS)}."
        )
        return None, report

    # Convert pollutant columns to numeric and profile missingness WITHOUT arbitrarily filling NaNs
    for pollutant in found_pollutants:
        usable_df[pollutant] = pd.to_numeric(usable_df[pollutant], errors="coerce")
        missing_count = int(usable_df[pollutant].isna().sum())
        missing_pct = (missing_count / len(usable_df)) * 100.0
        report.missing_pollutant_counts[pollutant] = missing_count
        report.missing_pollutant_percentages[pollutant] = round(missing_pct, 2)

        if missing_pct > 25.0:
            report.warnings.append(
                f"Pollutant '{pollutant}' has {missing_pct:.1f}% missing values. "
                "Baseline and anomaly reliability will be degraded."
            )

    # 7. Profile weather variables
    found_weather = [w for w in CANONICAL_WEATHER if w in usable_df.columns]
    report.available_weather = found_weather
    for w in found_weather:
        if w != "wd":
            usable_df[w] = pd.to_numeric(usable_df[w], errors="coerce")

    # Sort chronologically by station and timestamp
    usable_df = usable_df.sort_values(by=["station", "timestamp"]).reset_index(drop=True)

    # Metadata attributes
    usable_df.attrs["is_synthetic"] = is_synthetic
    usable_df.attrs["report"] = report

    report.is_valid = True
    return usable_df, report
