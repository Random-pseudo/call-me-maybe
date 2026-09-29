"""JSON schema parser for translating function definitions into rules for the FSA.

This module converts Pydantic FunctionDefinition models into structured schemas
consumed by the Finite State Automaton.
"""

from typing import Dict, List
from src.models.functions import FunctionDefinition


class ParsedFunctionSchema:
    """Represents a simplified, structured view of a function's schema for constrained decoding.

    Attributes:
        name: Name of the function.
        parameters: Ordered mapping of parameter names to their data types.
    """

    def __init__(self, name: str, parameters: Dict[str, str]) -> None:
        """Initialize the parsed function schema.

        Args:
            name: Name of the function.
            parameters: Mapping of parameter name to type string (e.g. 'number', 'string').
        """
        self.name = name
        self.parameters = parameters

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return f"ParsedFunctionSchema(name={self.name!r}, parameters={self.parameters!r})"


def parse_function_definitions(definitions: List[FunctionDefinition]) -> Dict[str, ParsedFunctionSchema]:
    """Translate a list of Pydantic FunctionDefinition objects into a lookup dictionary of schemas.

    Args:
        definitions: The list of validated FunctionDefinition models.

    Returns:
        A dictionary mapping each function name to its ParsedFunctionSchema.
    """
    schemas: Dict[str, ParsedFunctionSchema] = {}
    for fn in definitions:
        param_types = {param_name: param_schema.type for param_name, param_schema in fn.parameters.items()}
        schemas[fn.name] = ParsedFunctionSchema(name=fn.name, parameters=param_types)
    return schemas
