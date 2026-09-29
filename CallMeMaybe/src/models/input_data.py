"""Pydantic models for the input test queries.

This module validates the list of natural language prompts according to the project specifications.
"""

from typing import List
from pydantic import BaseModel, Field, RootModel


class PromptItem(BaseModel):
    """Model representing a single test prompt item.

    Attributes:
        prompt: The natural language request string.
    """

    prompt: str = Field(..., min_length=1)


class QueryList(RootModel[List[PromptItem]]):
    """Model representing the list of input queries.

    This acts as a strict validator for a JSON array of prompt objects.
    """

    pass
