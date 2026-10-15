"""Pydantic models for the generated output results.

This module ensures the final JSON output matches the expected format
specified in the evaluation criteria.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field, RootModel


class FunctionCallResult(BaseModel):
    """Model representing a single generated function call result.

    Attributes:
        prompt: The original user prompt.
        name: The name of the function selected by the LLM.
        parameters: The parsed parameters extracted by the LLM
            conforming to the function schema.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class FunctionCallResultList(RootModel[List[FunctionCallResult]]):
    """Model representing the entire output JSON file."""

    pass
