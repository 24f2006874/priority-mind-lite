from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    """Environment observation returned to the acting agent."""

    model_config = ConfigDict(extra="forbid")

    ticket_text: str = Field(..., description="Customer message content")
    sentiment: float = Field(..., ge=-1.0, le=1.0, description="Customer sentiment from -1 to 1")
    category: Optional[Literal["billing", "technical", "general", "complaint"]] = None
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = None
    attempts: int = Field(default=0, ge=0)
    resolved: bool = Field(default=False)


class Action(BaseModel):
    """Action chosen by the agent."""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["categorize", "prioritize", "respond", "escalate", "resolve"]
    content: Optional[str] = Field(default=None, description="Content for categorize or respond actions")
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = Field(
        default=None, description="Priority level for prioritize action"
    )


class Reward(BaseModel):
    """Normalized reward with interpretable partial signals."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(..., ge=0.0, le=1.0, description="Normalized reward score between 0 and 1")
    reasoning: str = Field(default="", description="Human-readable explanation for the reward")
    partial_signals: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional reward signals like empathy, efficiency, strategy",
    )
