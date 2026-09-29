"""Pydantic models for function definitions.

This module defines the schema for the functions available to the LLM.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class ParameterSchema(BaseModel):
    """Schema definition for a single parameter.

    Attributes:
        type: The data type of the parameter (e.g., "number", "string", "boolean").
        description: An optional description of the parameter.
    """

    type: str
    description: Optional[str] = None


class ReturnSchema(BaseModel):
    """Schema definition for the return value of a function.

    Attributes:
        type: The data type of the return value.
    """

    type: str


class FunctionDefinition(BaseModel):
    """Model representing a single function definition.

    Attributes:
        name: The precise name of the function.
        description: What the function does.
        parameters: A dictionary mapping parameter names to their schemas.
        returns: The schema of the return value.
    """

    name: str = Field(..., min_length=1)
    description: str
    parameters: Dict[str, ParameterSchema] = Field(default_factory=dict)
    returns: ReturnSchema
