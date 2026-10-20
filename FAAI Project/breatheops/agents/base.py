"""
Base agent abstractions and structured result types for BreatheOps AI Phase 2.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(str, Enum):
    """Execution status for an investigation agent."""
    IDLE = "IDLE"
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass
class AgentResult:
    """Standardized structured output returned by all BreatheOps AI agents."""
    agent_name: str
    status: AgentStatus
    summary: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    next_action: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "summary": self.summary,
            "evidence": self.evidence,
            "warnings": self.warnings,
            "limitations": self.limitations,
            "next_action": self.next_action,
            "timestamp": self.timestamp,
        }


class BaseAgent(ABC):
    """Abstract base class for all specialized investigation agents."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, context: Any) -> AgentResult:
        """Execute the agent's specialized task given an investigation context."""
        pass
