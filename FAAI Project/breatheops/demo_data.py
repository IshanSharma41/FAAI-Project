"""
Synthetic demo data generation for BreatheOps AI.

Generates reproducible, realistic multi-station air quality and meteorological
time-series with at least one deliberately injected pollution spike, designed for
educational and faculty demonstration.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

DEFAULT_STATIONS = ["Aotizhongxin", "Changping", "Dongsi", "Wanliu"]
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER_VARS = ["TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"]
WIND_DIRECTIONS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def generate_demo_data(
    seed: int = 42,
    days: int = 14,
    stations: Optional[List[str]] = None,
    inject_spike: bool = True,
    spike_station: str = "Aotizhongxin",
    start_date: str = "2023-01-01 00:00:00",
) -> pd.DataFrame:
    """Generate reproducible synthetic air-quality dataset.

    Parameters:
        seed: Random seed for deterministic generation.
        days: Number of days of hourly observations (default: 14 days = 336 hours).
        stations: List of monitoring station names.
        inject_spike: Whether to deliberately inject a pollution event.
        spike_station: Station where the spike should be injected.
        start_date: Start timestamp for the time series.

    Returns:
        pd.DataFrame formatted similarly to the Beijing Multi-Site PRSA dataset.
    """
    rng = np.random.default_rng(seed)
    if stations is None:
        stations = DEFAULT_STATIONS

    timestamps = pd.date_range(start=start_date, periods=days * 24, freq="h")
    total_hours = len(timestamps)

    frames = []

    for station_idx, station in enumerate(stations):
        # Station-specific baseline multiplier for natural diversity
        station_bias = 0.85 + (station_idx * 0.1)

        # Diurnal and multi-day cyclic variations
        hours = np.arange(total_hours)
        diurnal_pm = np.sin((hours % 24 - 7) * (2 * np.pi / 24)) * 15.0  # Morning rush / evening rise
        diurnal_temp = np.sin((hours % 24 - 14) * (2 * np.pi / 24)) * 6.0  # Peak in afternoon
        multi_day_wave = np.sin(hours * (2 * np.pi / (24 * 4))) * 12.0  # 4-day weather wave

        # Background pollutant levels with noise
        pm25 = np.clip(
            (45.0 * station_bias) + diurnal_pm + multi_day_wave + rng.normal(0, 8.0, total_hours),
            a_min=3.0,
            a_max=160.0,
        )
        pm10 = np.clip(
            pm25 * 1.5 + rng.normal(0, 10.0, total_hours),
            a_min=6.0,
            a_max=250.0,
        )
        so2 = np.clip(
            (12.0 * station_bias) + rng.normal(0, 3.5, total_hours),
            a_min=1.0,
            a_max=60.0,
        )
        no2 = np.clip(
            (35.0 * station_bias) + diurnal_pm * 0.8 + rng.normal(0, 6.0, total_hours),
            a_min=5.0,
            a_max=120.0,
        )
        co = np.clip(
            (700.0 * station_bias) + diurnal_pm * 15.0 + rng.normal(0, 100.0, total_hours),
            a_min=150.0,
            a_max=2400.0,
        )
        # Ozone peaks in the sunny afternoon inversely with NO2
        o3 = np.clip(
            (40.0 * station_bias) + diurnal_temp * 4.0 + rng.normal(0, 8.0, total_hours),
            a_min=2.0,
            a_max=180.0,
        )

        # Meteorological variables
        temp = -2.0 + diurnal_temp + multi_day_wave * 0.3 + rng.normal(0, 1.2, total_hours)
        pres = 1022.0 - (multi_day_wave * 0.5) + rng.normal(0, 1.5, total_hours)
        dewp = temp - 14.0 + rng.normal(0, 2.0, total_hours)
        rain = np.where(rng.random(total_hours) > 0.94, rng.exponential(1.5, total_hours), 0.0)
        wspm = np.clip(2.5 + rng.normal(0, 0.9, total_hours), a_min=0.3, a_max=8.5)
        wd_indices = rng.integers(0, len(WIND_DIRECTIONS), total_hours)
        wd = [WIND_DIRECTIONS[i] for i in wd_indices]

        # Injected deliberate pollution event
        if inject_spike and station == spike_station and total_hours >= 24:
            # Place spike comfortably within the timeline
            spike_len = min(10, max(4, total_hours // 8))
            spike_start = min(160, max(6, int(total_hours * 0.5) - (spike_len // 2)))
            if spike_start + spike_len > total_hours:
                spike_start = total_hours - spike_len
            spike_end = spike_start + spike_len

            # Severe PM spike: peaking at ~420 µg/m³
            spike_profile = np.sin(np.linspace(0.2, np.pi - 0.2, spike_len))
            pm25[spike_start:spike_end] += spike_profile * 340.0
            pm10[spike_start:spike_end] += spike_profile * 420.0
            no2[spike_start:spike_end] += spike_profile * 80.0
            co[spike_start:spike_end] += spike_profile * 1800.0
            so2[spike_start:spike_end] += spike_profile * 45.0

            # Stagnant atmospheric conditions during spike: low wind speed
            wspm[spike_start:spike_end] = np.clip(wspm[spike_start:spike_end] * 0.25, 0.2, 0.8)

        # Introduce small, realistic missing value rate (~1.5%) across pollutants
        for col_data in [pm25, pm10, so2, no2, co, o3]:
            missing_mask = rng.random(total_hours) < 0.015
            # Do not erase the injected spike peak so the spike remains detectable in tests
            if inject_spike and station == spike_station and total_hours >= 24:
                missing_mask[spike_start:spike_end] = False
            col_data[missing_mask] = np.nan

        station_df = pd.DataFrame(
            {
                "year": timestamps.year,
                "month": timestamps.month,
                "day": timestamps.day,
                "hour": timestamps.hour,
                "PM2.5": np.round(pm25, 1),
                "PM10": np.round(pm10, 1),
                "SO2": np.round(so2, 1),
                "NO2": np.round(no2, 1),
                "CO": np.round(co, 1),
                "O3": np.round(o3, 1),
                "TEMP": np.round(temp, 1),
                "PRES": np.round(pres, 1),
                "DEWP": np.round(dewp, 1),
                "RAIN": np.round(rain, 1),
                "wd": wd,
                "WSPM": np.round(wspm, 1),
                "station": station,
                "timestamp": timestamps,
            }
        )
        frames.append(station_df)

    combined_df = pd.concat(frames, ignore_index=True)
    # Add explicit synthetic flag attribute
    combined_df.attrs["is_synthetic"] = True
    combined_df.attrs["injected_spike_station"] = spike_station if inject_spike else None
    return combined_df
