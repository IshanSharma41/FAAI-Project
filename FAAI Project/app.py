"""
BreatheOps AI — Air Quality Investigation Centre
Phase 3: Streamlined Faculty Demonstration, Usability & Robust Multi-Agent Advisory

An explainable multi-agent system coordinating Planner, Data Quality, Spike Detection,
Multi-Station Context, Response Planning, and Critic & Safety Agents.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from breatheops.analysis import POLLUTANT_UNITS, AnomalyDetectionResult
from breatheops.data_loader import (
    CANONICAL_POLLUTANTS,
    DataValidationReport,
    load_air_quality_csv,
)
from breatheops.demo_data import generate_demo_data
from breatheops.orchestrator import (
    InvestigationContext,
    InvestigationOrchestrator,
    InvestigationReport,
)

# Page configuration
st.set_page_config(
    page_title="BreatheOps AI — Air Quality Investigation Centre",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for faculty-grade presentation
st.markdown(
    """
    <style>
    .main-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        font-weight: 500;
        color: #0284C7;
        margin-bottom: 1.2rem;
        letter-spacing: 0.3px;
    }
    .badge-synthetic {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #FCD34D;
        margin-bottom: 1rem;
    }
    .badge-real {
        display: inline-block;
        background-color: #ECFDF5;
        color: #065F46;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #A7F3D0;
        margin-bottom: 1rem;
    }
    .badge-status {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .badge-completed { background-color: #DEF7EC; color: #03543F; }
    .badge-warning { background-color: #FDF6B2; color: #723B13; }
    .badge-skipped { background-color: #E5E7EB; color: #374151; }
    .badge-failed { background-color: #FDE8E8; color: #9B1C1C; }
    .badge-priority {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-right: 10px;
    }
    .priority-high { background-color: #FEE2E2; color: #991B1B; border: 1px solid #F87171; }
    .priority-elevated { background-color: #FEF3C7; color: #92400E; border: 1px solid #FBBF24; }
    .priority-routine { background-color: #E0E7FF; color: #3730A3; border: 1px solid #818CF8; }
    .priority-audit { background-color: #F3E8FF; color: #6B21A8; border: 1px solid #C084FC; }
    .priority-provisional { background-color: #F1F5F9; color: #475569; border: 1px solid #94A3B8; }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .card-title {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .card-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
    }
    .finding-box {
        background-color: #F0F9FF;
        border-left: 5px solid #0284C7;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0 20px 0;
        font-size: 1.02rem;
        line-height: 1.6;
        color: #0C4A6E;
    }
    .agent-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .disclaimer-box {
        background-color: #F8FAFC;
        border: 1px dashed #CBD5E1;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_demo_data(seed: int = 42, days: int = 14) -> pd.DataFrame:
    """Generate and cache reproducible synthetic demonstration dataset."""
    return generate_demo_data(seed=seed, days=days, inject_spike=True, spike_station="Aotizhongxin")


def get_available_local_samples() -> list[str]:
    """Check for local sample datasets in PRSA directory."""
    sample_dir = Path("PRSA_Data_20130301-20170228")
    if sample_dir.exists() and sample_dir.is_dir():
        files = sorted(list(sample_dir.glob("*.csv")))
        return [f.name for f in files]
    return []


def create_time_series_chart(
    df: pd.DataFrame,
    result: Optional[AnomalyDetectionResult],
    target_pollutant: str,
    target_station: str,
) -> go.Figure:
    """Create interactive Plotly chart with readings, baseline, threshold, and highlighted anomalies."""
    unit = POLLUTANT_UNITS.get(target_pollutant, "µg/m³")
    fig = go.Figure()

    if target_pollutant in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[target_pollutant],
                mode="lines",
                name=f"{target_pollutant} Observed ({unit})",
                line=dict(color="#2563EB", width=1.75),
                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Reading: %{y:.1f} " + unit + "<extra></extra>",
            )
        )

    if result:
        # Baseline Median line
        if result.baseline_median is not None:
            fig.add_hline(
                y=result.baseline_median,
                line_dash="dot",
                line_color="#10B981",
                line_width=1.8,
                annotation_text=f"Baseline Median ({result.baseline_median:.1f} {unit})",
                annotation_position="bottom right",
                annotation_font_size=11,
                annotation_font_color="#065F46",
            )

        # Detection Threshold line
        if result.threshold_value is not None:
            fig.add_hline(
                y=result.threshold_value,
                line_dash="dash",
                line_color="#EF4444",
                line_width=2.0,
                annotation_text=f"Threshold: {result.threshold_value:.1f} {unit} ({result.threshold_multiplier}× MAD)",
                annotation_position="top right",
                annotation_font_size=11,
                annotation_font_color="#991B1B",
            )

        # Highlighted anomalous points
        if result.anomaly_count > 0 and not result.anomalies_df.empty:
            anom = result.anomalies_df
            custom_text = [
                f"<b>ANOMALY DETECTED</b><br>Time: {ts}<br>Observed: {val:.1f} {unit}<br>"
                f"Baseline: {result.baseline_median:.1f} {unit}<br>Diff: +{val - result.baseline_median:.1f} {unit}<br>"
                f"Z-Score: {z:.2f}"
                for ts, val, z in zip(anom["timestamp"], anom[target_pollutant], anom.get("robust_z_score", [0] * len(anom)))
            ]
            fig.add_trace(
                go.Scatter(
                    x=anom["timestamp"],
                    y=anom[target_pollutant],
                    mode="markers",
                    name=f"Anomalies ({result.anomaly_count})",
                    marker=dict(
                        color="#DC2626",
                        size=9,
                        symbol="circle",
                        line=dict(color="#FFFFFF", width=1.5),
                    ),
                    text=custom_text,
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        title=dict(
            text=f"Time Series: {target_pollutant} at Station '{target_station}' with Anomaly Detection",
            font=dict(size=15, color="#1E293B"),
        ),
        xaxis=dict(title="Observation Timestamp", gridcolor="#F1F5F9", showgrid=True),
        yaxis=dict(title=f"Concentration ({unit})", gridcolor="#F1F5F9", showgrid=True),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=55, b=40),
        height=390,
    )

    return fig


def main():
    """Main application layout and control flow."""
    # App Title & Subtitle
    st.markdown('<div class="main-header">BreatheOps AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Air Quality Event Investigation and Advisory System &mdash; Demonstration Centre</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Data Source & Demonstration Configuration
    st.sidebar.header("📁 1. Data Source Selection")

    local_samples = get_available_local_samples()
    source_options = ["Built-in Demo Data (Synthetic)"]
    if local_samples:
        source_options.append("Beijing Multi-Site Sample (Local)")
    source_options.append("Upload CSV File")

    data_source = st.sidebar.radio(
        "Select Input Source:",
        options=source_options,
        index=0,
        help="Built-in demo is fully reproducible and offline with an injected spike on station Aotizhongxin.",
    )

    loaded_df: Optional[pd.DataFrame] = None
    validation_report: Optional[DataValidationReport] = None
    is_synthetic: bool = False
    source_label: str = ""

    try:
        if data_source == "Built-in Demo Data (Synthetic)":
            is_synthetic = True
            source_label = "Synthetic Demo Dataset (Reproducible, Seed=42)"
            loaded_df = get_cached_demo_data()
            validation_report = DataValidationReport(
                is_valid=True,
                total_rows=len(loaded_df),
                usable_rows=len(loaded_df),
                available_stations=sorted(loaded_df["station"].unique()),
                available_pollutants=[p for p in CANONICAL_POLLUTANTS if p in loaded_df.columns],
                available_weather=["TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"],
                is_synthetic=True,
            )

        elif data_source == "Beijing Multi-Site Sample (Local)":
            chosen_file = st.sidebar.selectbox("Choose Reference Station File:", options=local_samples)
            sample_path = Path("PRSA_Data_20130301-20170228") / chosen_file
            source_label = f"Real Dataset: {chosen_file}"
            is_synthetic = False
            with st.spinner("Loading reference observation records..."):
                loaded_df, validation_report = load_air_quality_csv(sample_path, is_synthetic=False)

        elif data_source == "Upload CSV File":
            uploaded_file = st.sidebar.file_uploader(
                "Upload Air Quality CSV",
                type=["csv"],
                help="Requires timestamp or (year, month, day, hour) and at least one pollutant column.",
            )
            if uploaded_file is not None:
                source_label = f"Uploaded File: {uploaded_file.name}"
                is_synthetic = False
                with st.spinner("Parsing and validating uploaded CSV..."):
                    loaded_df, validation_report = load_air_quality_csv(uploaded_file, is_synthetic=False)
            else:
                st.info("👈 Please upload a CSV file in the sidebar to proceed, or choose 'Built-in Demo Data'.")
                st.stop()

    except Exception as exc:
        st.error(f"❌ Error loading dataset: {str(exc)}")
        st.stop()

    if validation_report is None or not validation_report.is_valid or loaded_df is None:
        st.error("❌ Data Ingestion Failed")
        if validation_report and validation_report.errors:
            for err in validation_report.errors:
                st.error(f"- {err}")
        st.info("💡 Requirements: CSV must contain either a 'timestamp' column or date components ('year', 'month', 'day', 'hour') and recognized pollutant columns (PM2.5, PM10, etc.).")
        st.stop()

    # Source Banner (Explicit Identification)
    if is_synthetic:
        st.markdown(
            '<div class="badge-synthetic">⚠️ SYNTHETIC DEMONSTRATION DATA &mdash; Injected Multi-Pollutant Spike on Station Aotizhongxin (Fixed Seed 42)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="badge-real">📊 REAL OBSERVATION DATA &mdash; {source_label} ({validation_report.usable_rows:,} usable observations)</div>',
            unsafe_allow_html=True,
        )

    # Sidebar: Target Selection
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 2. Target Station & Pollutant")

    available_stations = validation_report.available_stations
    selected_station = st.sidebar.selectbox(
        "Monitoring Station:",
        options=available_stations,
        index=0 if available_stations else 0,
        help="Select environmental monitoring station to evaluate.",
    )

    available_pollutants = validation_report.available_pollutants
    default_pollutant_idx = available_pollutants.index("PM2.5") if "PM2.5" in available_pollutants else 0
    selected_pollutant = st.sidebar.selectbox(
        "Target Pollutant:",
        options=available_pollutants,
        index=default_pollutant_idx,
        help="Select target pollutant for baseline and anomaly detection.",
    )

    # Sidebar: Observation Timeframe Filter
    st.sidebar.markdown("---")
    st.sidebar.header("⏱️ 3. Observation Period")

    station_records = loaded_df[loaded_df["station"] == selected_station]
    if not station_records.empty and "timestamp" in station_records.columns:
        min_ts = station_records["timestamp"].min()
        max_ts = station_records["timestamp"].max()
        if pd.notna(min_ts) and pd.notna(max_ts) and min_ts != max_ts:
            min_date = min_ts.date()
            max_date = max_ts.date()
            date_selection = st.sidebar.date_input(
                "Filter Timeframe:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(date_selection, (list, tuple)) and len(date_selection) == 2:
                start_date, end_date = date_selection
                mask = (loaded_df["timestamp"].dt.date >= start_date) & (loaded_df["timestamp"].dt.date <= end_date)
                analysis_df = loaded_df[mask].copy()
            else:
                analysis_df = loaded_df.copy()
        else:
            analysis_df = loaded_df.copy()
    else:
        analysis_df = loaded_df.copy()

    if analysis_df.empty:
        st.warning(f"⚠️ No observations found within the selected timeframe. Please widen the date interval in the sidebar.")
        st.stop()

    # Sidebar: Sensitivity
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 4. Detection Parameters")
    k_multiplier = st.sidebar.slider(
        "Sensitivity Threshold (k × MAD):",
        min_value=1.5,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Defines threshold: Median + k × (1.4826 × MAD). Standard default is 3.0 (3-sigma equivalent).",
    )

    # Prominent Primary Action Buttons
    st.sidebar.markdown("---")
    col_run, col_reset = st.sidebar.columns([3, 2])
    with col_run:
        run_btn = st.button("🚀 Run Investigation", type="primary", use_container_width=True)
    with col_reset:
        reset_btn = st.button("🔄 Reset", use_container_width=True, help="Clear cached results and reset investigation to initial state.")

    # Reset handling
    if reset_btn:
        for k in ["current_report", "current_context", "orchestration_key"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # Run Investigation Orchestrator
    orchestration_key = f"{selected_station}_{selected_pollutant}_{len(analysis_df)}_{k_multiplier}"
    if "current_report" not in st.session_state or run_btn or st.session_state.get("orchestration_key") != orchestration_key:
        with st.spinner("Executing 6 specialized agents: Planning, Data Quality, Detection, Spatial, Response, Critic..."):
            try:
                orchestrator = InvestigationOrchestrator()
                report, context = orchestrator.run_investigation(
                    df=analysis_df,
                    target_station=selected_station,
                    target_pollutant=selected_pollutant,
                    k_multiplier=k_multiplier,
                    is_synthetic=is_synthetic,
                    dataset_source_label=source_label,
                    validation_report=validation_report,
                )
                st.session_state["current_report"] = report
                st.session_state["current_context"] = context
                st.session_state["orchestration_key"] = orchestration_key
            except Exception as exc:
                st.error(f"❌ Error during multi-agent orchestration: {str(exc)}")
                st.stop()
    else:
        report: InvestigationReport = st.session_state["current_report"]
        context: InvestigationContext = st.session_state["current_context"]

    # -------------------------------------------------------------
    # 1. Investigation Overview and Key Results
    # -------------------------------------------------------------
    st.markdown("### 1. Investigation Overview & Key Results")

    priority_class_map = {
        "HIGH_PRIORITY_INVESTIGATION": "priority-high",
        "ELEVATED_INVESTIGATION": "priority-elevated",
        "ROUTINE_MONITORING": "priority-routine",
        "DATA_INTEGRITY_AUDIT": "priority-audit",
        "PROVISIONAL_OBSERVATION": "priority-provisional",
    }
    p_class = priority_class_map.get(report.assigned_priority, "priority-routine")

    v_badge = (
        '<span style="background-color: #DEF7EC; color: #03543F; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">🛡️ VALIDATED</span>'
        if report.validation_status == "VALIDATED"
        else '<span style="background-color: #FEF08A; color: #854D0E; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">⚠️ REQUIRES HUMAN REVIEW</span>'
    )

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
            <div>
                <span class="badge-priority {p_class}">Priority: {report.assigned_priority.replace('_', ' ')}</span>
                {v_badge}
            </div>
            <span style="font-size: 0.85rem; color: #64748B;">Investigation ID: <code>{report.investigation_id}</code></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Executive Plain-Language Advisory Callout
    st.markdown(f'<div class="finding-box">{report.plain_language_advisory}</div>', unsafe_allow_html=True)

    # Summary Metrics Row
    unit = POLLUTANT_UNITS.get(selected_pollutant, "µg/m³")
    spike_ev = context.spike_evidence

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Observations</div>
                <div class="card-value">{spike_ev.get('valid_observations', 0):,}</div>
                <small style="color: #64748B;">Station '{selected_station}'</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        med_str = f"{report.baseline_median:.1f} {unit}" if report.baseline_median is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Baseline Median</div>
                <div class="card-value">{med_str}</div>
                <small style="color: #64748B;">Historical Station Norm</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        thresh_str = f"{report.threshold_value:.1f} {unit}" if report.threshold_value is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Threshold</div>
                <div class="card-value">{thresh_str}</div>
                <small style="color: #64748B;">{k_multiplier}× Scaled MAD</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        anom_color = "#DC2626" if report.anomaly_count > 0 else "#10B981"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Anomalies Detected</div>
                <div class="card-value" style="color: {anom_color};">{report.anomaly_count}</div>
                <small style="color: #64748B;">{'Spike Event Identified' if report.anomaly_count > 0 else 'Nominal Baseline'}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m5:
        spatial_lbl = context.multi_station_evidence.get("spatial_classification", "SINGLE_STATION")
        if "ISOLATED" in spatial_lbl:
            disp_badge = "📍 Isolated Local"
            disp_color = "#D97706"
        elif "REGIONAL" in spatial_lbl:
            disp_badge = "🌐 Co-Occurring"
            disp_color = "#2563EB"
        elif "UNIFORM" in spatial_lbl:
            disp_badge = "✅ Nominal Regional"
            disp_color = "#059669"
        else:
            disp_badge = "ℹ️ Single Station"
            disp_color = "#64748B"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Spatial Context</div>
                <div class="card-value" style="color: {disp_color}; font-size: 1.25rem;">{disp_badge}</div>
                <small style="color: #64748B;">Cross-Station Comparison</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin: 25px 0 20px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. Pollution Trends and Detected Anomalies
    # -------------------------------------------------------------
    st.markdown("### 2. Pollution Trends & Detected Anomalies")

    station_plot_df = analysis_df[analysis_df["station"] == selected_station] if "station" in analysis_df.columns else analysis_df
    if not station_plot_df.empty and "timestamp" in station_plot_df.columns:
        fig = create_time_series_chart(
            station_plot_df,
            context.spike_detection_result,
            selected_pollutant,
            selected_station,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient observation records to plot time series.")

    # Anomalies Breakdown Table
    if context.spike_detection_result and context.spike_detection_result.anomaly_count > 0:
        with st.expander(f"📋 View Detected Anomalous Readings ({context.spike_detection_result.anomaly_count} observations)", expanded=False):
            anom_df = context.spike_detection_result.anomalies_df.copy()
            if "timestamp" in anom_df.columns:
                anom_df["timestamp"] = anom_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(anom_df, use_container_width=True, hide_index=True)
            st.download_button(
                label="📥 Export Detected Anomalies (CSV)",
                data=anom_df.to_csv(index=False).encode("utf-8"),
                file_name=f"breatheops_anomalies_{selected_station}_{selected_pollutant}.csv",
                mime="text/csv",
            )
    else:
        st.success(f"✅ No readings exceeded the statistical anomaly threshold of {report.threshold_value} {unit}.")

    st.markdown("<hr style='margin: 25px 0 20px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. Data-Quality & Cross-Station Evidence
    # -------------------------------------------------------------
    st.markdown("### 3. Data-Quality & Cross-Station Evidence")

    ev_col1, ev_col2 = st.columns([1, 1])

    with ev_col1:
        st.markdown("#### 📊 Data Quality & Sensor Profiling")
        st.markdown(f"**Data Audit Summary**: {report.data_quality_findings}")
        dq_ev = context.data_quality_evidence
        st.markdown(
            f"""
            - **Total Records Analyzed**: {dq_ev.get('total_rows', 0):,}
            - **Usable Timestamps**: {dq_ev.get('usable_rows', 0):,}
            - **Invalid Timestamps Filtered**: {dq_ev.get('invalid_timestamps', 0)}
            - **Duplicate Observations Deduplicated**: {dq_ev.get('duplicate_records', 0)}
            - **Missing Rate ({selected_pollutant})**: {dq_ev.get('missing_target_percentage', 0.0):.1f}%
            """
        )

    with ev_col2:
        st.markdown("#### 🌐 Cross-Station Spatial & Weather Context")
        st.markdown(f"**Spatial Finding**: {report.cross_station_finding}")
        st.markdown(f"**Meteorological Context**: {report.weather_context}")

        multi_ev = context.multi_station_evidence
        if multi_ev.get("cross_station_available") and multi_ev.get("station_window_stats"):
            st_stats = multi_ev.get("station_window_stats", {})
            stat_rows = [
                {
                    "Station": stn,
                    f"Mean {selected_pollutant} ({unit})": data.get("mean"),
                    f"Peak {selected_pollutant} ({unit})": data.get("max"),
                    "Obs": data.get("count"),
                }
                for stn, data in st_stats.items()
            ]
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    st.caption("ℹ️ *Notice: Spatial comparisons and meteorological readings provide environmental context only and do not establish causation.*")

    st.markdown("<hr style='margin: 25px 0 20px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. Agent Execution Log
    # -------------------------------------------------------------
    st.markdown("### 4. Agent Execution Log & Workflow Tracking")
    st.caption("Inspect the explicit execution sequence, status, and empirical findings passed across the multi-agent orchestrator.")

    if context.plan:
        with st.expander("🧭 View Investigation Plan & Execution Tracking", expanded=True):
            st.markdown(f"**Plan ID**: `{context.plan.plan_id}` | Target: `{context.plan.target_station}` ({context.plan.target_pollutant})")
            for step in context.plan.planned_steps:
                badge_style = {
                    "COMPLETED": "badge-completed",
                    "WARNING": "badge-warning",
                    "SKIPPED": "badge-skipped",
                    "FAILED": "badge-failed",
                }.get(step.status.value, "badge-skipped")
                st.markdown(
                    f"""
                    <div class="agent-card">
                        <span class="badge-status {badge_style}">{step.status.value}</span>
                        <b>Step {step.step_id}: {step.agent_name}</b> &mdash; <i>{step.action}</i><br>
                        <span style="font-size: 0.9rem; color: #475569;">{step.description}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("🔬 Individual Agent Detailed Artifacts & Raw Evidence", expanded=False):
        for agent_log in report.agent_execution_log:
            agent_name = agent_log.get("agent_name", "Agent")
            status = agent_log.get("status", "COMPLETED")
            summary = agent_log.get("summary", "")
            st.markdown(f"**{agent_name}** `[{status}]`: {summary}")
            if agent_log.get("warnings"):
                for w in agent_log["warnings"]:
                    st.warning(f"⚠️ {w}")
            if agent_log.get("limitations"):
                for l in agent_log["limitations"]:
                    st.info(f"ℹ️ {l}")
            st.json(agent_log.get("evidence", {}), expanded=False)

    st.markdown("<hr style='margin: 25px 0 20px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 5. Recommended Actions & Validation Status
    # -------------------------------------------------------------
    st.markdown("### 5. Recommended Actions & Validation Status")

    act_col1, act_col2 = st.columns([1, 1])

    with act_col1:
        st.markdown("#### 🚨 Recommended Follow-Up Actions")
        st.markdown(f"**Assigned Priority**: `{report.assigned_priority}`")
        st.markdown(f"**Rationale**: {report.priority_reasoning}")

        if report.recommended_actions:
            for rec in report.recommended_actions:
                st.markdown(
                    f"""
                    <div style="background-color: #F8FAFC; border-left: 4px solid #3B82F6; padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 10px;">
                        <span style="font-size: 0.78rem; font-weight: 700; color: #2563EB;">[{rec.get('action_id')}] {rec.get('target_team')} | Urgency: {rec.get('urgency')}</span>
                        <h4 style="margin: 4px 0 6px 0; color: #0F172A; font-size: 1.05rem;">{rec.get('title')}</h4>
                        <p style="margin: 0 0 4px 0; font-size: 0.9rem; color: #334155;"><b>Rationale</b>: {rec.get('rationale')}</p>
                        <p style="margin: 0; font-size: 0.85rem; color: #64748B;"><b>Action Details</b>: {rec.get('details')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No specialized follow-up actions required; continue routine background monitoring.")

    with act_col2:
        st.markdown("#### 🛡️ Critic & Safety Agent Certification")
        st.markdown(f"**Certification Result**: `{report.validation_status}`")
        st.markdown(f"{report.critic_audit_summary}")

        critic_ev = context.critic_evidence
        checks = critic_ev.get("audit_checks", [])
        if checks:
            st.markdown("**Empirical Audit Checks Conducted:**")
            for c in checks:
                icon = "✅" if c.get("passed") else "⚠️"
                st.markdown(f"- {icon} **{c.get('name')}**: {c.get('details')}")

        if critic_ev.get("modifications_made"):
            for mod in critic_ev["modifications_made"]:
                st.info(f"🛡️ Guardrail Action: {mod}")

    st.markdown("<hr style='margin: 25px 0 20px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 6. Final Investigation Report
    # -------------------------------------------------------------
    st.markdown("### 6. Final Evidence-Based Investigation Report")
    st.caption("Complete synthesized dossier based strictly on verified numerical findings, with export options in Markdown and Plain Text.")

    dl_col1, dl_col2, _ = st.columns([2, 2, 4])
    with dl_col1:
        st.download_button(
            label="📥 Download Report (.md)",
            data=report.markdown_report.encode("utf-8"),
            file_name=f"BreatheOps_Report_{report.investigation_id}.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label="📄 Download Report (.txt)",
            data=report.plain_text_report.encode("utf-8"),
            file_name=f"BreatheOps_Report_{report.investigation_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("📄 Read Full In-App Formatted Report", expanded=True):
        st.markdown(report.markdown_report)

    # Footer Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            ⚠️ <b>Academic & Non-Regulatory Disclaimer</b>: Statistical anomaly detection identifies measurements that deviate
            significantly from station baseline norms. This analysis does not represent an official regulatory violation, statutory
            compliance determination, or official environmental emergency classification.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
