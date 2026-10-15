"""Pydantic models for function definitions.

This module defines the schema for the functions available to the LLM,
validating function metadata, parameters, and return types.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, RootModel


class ParameterSchema(BaseModel):
    """Schema definition for a single parameter.

    Attributes:
        type: The data type of the parameter
            (e.g., "number", "string", "boolean").
        description: An optional human-readable description of the parameter.
        enum: An optional list of predefined allowed values
            for enum constraints.
        default: An optional default value for the parameter.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["string", "number", "integer", "boolean"]


class FunctionDefinition(BaseModel):
    """Model representing a single function definition.

    Attributes:
        name: The precise name of the function.
        description: Human-readable description of what the function does.
        parameters: A dictionary mapping parameter names to their schemas.
        returns: The schema of the return value.
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., min_length=1)
    description: str
    parameters: Dict[str, ParameterSchema] = Field(default_factory=dict)
    returns: Optional[ParameterSchema] = None


class FunctionsDefinitionList(RootModel[List[FunctionDefinition]]):
    """Model representing a validated list of function definitions."""

    pass
