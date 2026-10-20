"""
BreatheOps AI - Air Quality Event Investigation and Advisory System
Phase 1: Analytical and Data Processing Foundation
Phase 2: Multi-Agent Investigation Workflow and Evidence-Based Report Generation
"""

from breatheops.data_loader import (
    load_air_quality_csv,
    normalize_column_name,
    normalize_dataframe_columns,
    DataValidationReport,
    CANONICAL_POLLUTANTS,
    CANONICAL_WEATHER,
)
from breatheops.demo_data import (
    generate_demo_data,
    DEFAULT_STATIONS,
    POLLUTANTS,
    WEATHER_VARS,
)
from breatheops.analysis import (
    detect_anomalies_mad,
    calculate_robust_baseline,
    AnomalyDetectionResult,
    POLLUTANT_UNITS,
)
from breatheops.agents import (
    AgentResult,
    AgentStatus,
    BaseAgent,
    InvestigationPlan,
    PlannerAgent,
    PlanStep,
    DataQualityAgent,
    PollutionSpikeDetectionAgent,
    MultiStationContextAgent,
    InvestigationPriority,
    RecommendedAction,
    ResponsePlanningAgent,
    AuditCheckItem,
    CriticSafetyAgent,
    ValidationStatus,
)
from breatheops.orchestrator import (
    InvestigationContext,
    InvestigationOrchestrator,
    InvestigationReport,
)

__all__ = [
    # Phase 1
    "load_air_quality_csv",
    "normalize_column_name",
    "normalize_dataframe_columns",
    "DataValidationReport",
    "CANONICAL_POLLUTANTS",
    "CANONICAL_WEATHER",
    "generate_demo_data",
    "DEFAULT_STATIONS",
    "POLLUTANTS",
    "WEATHER_VARS",
    "detect_anomalies_mad",
    "calculate_robust_baseline",
    "AnomalyDetectionResult",
    "POLLUTANT_UNITS",
    # Phase 2 Agents & Orchestrator
    "AgentResult",
    "AgentStatus",
    "BaseAgent",
    "InvestigationPlan",
    "PlannerAgent",
    "PlanStep",
    "DataQualityAgent",
    "PollutionSpikeDetectionAgent",
    "MultiStationContextAgent",
    "InvestigationPriority",
    "RecommendedAction",
    "ResponsePlanningAgent",
    "AuditCheckItem",
    "CriticSafetyAgent",
    "ValidationStatus",
    "InvestigationContext",
    "InvestigationOrchestrator",
    "InvestigationReport",
]

__version__ = "2.0.0"
