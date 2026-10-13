"""JSON schema parser for translating function definitions into FSA rules.

This module converts Pydantic FunctionDefinition models into structured schemas
consumed by the Finite State Automaton and Generation pipeline.
"""

from typing import Any, Dict, List, Optional, Literal
from src.models.functions import FunctionDefinition


class ParsedParameterSchema:
    """Represents a validated parameter schema.

    Attributes:
        name: Name of the parameter.
        type: Data type string (e.g. 'number', 'string', 'boolean').
        description: Description of the parameter.
        enum: Predefined allowed values for enum constraints.
        default: Default value if optional.
    """

    def __init__(
        self,
        name: str,
        param_type: Literal["string", "number", "integer", "boolean"],
        description: Optional[str] = None,
        enum: Optional[List[Any]] = None,
        default: Optional[Any] = None,
    ) -> None:
        """Initialize the parsed parameter schema.

        Args:
            name: Name of the parameter.
            param_type: Data type string.
            description: Parameter description.
            enum: Optional allowed values.
            default: Optional default value.
        """
        self.name = name
        self.type = param_type.lower()
        self.description = description
        self.enum = enum
        self.default = default

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return (
            f"ParsedParameterSchema(name={self.name!r}, type={self.type!r}, "
            f"enum={self.enum!r}, default={self.default!r})"
        )


class ParsedFunctionSchema:
    """Represents a structured view of a function schema for decoding.

    Attributes:
        name: Name of the function.
        description: Description of what the function does.
        parameters: Ordered mapping of parameter names to schema.
        returns_type: Type of return value.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, ParsedParameterSchema],
        returns_type: Optional[str] = None,
    ) -> None:
        """Initialize the parsed function schema.

        Args:
            name: Name of the function.
            description: Function description.
            parameters: Mapping of parameter name to its ParsedParameterSchema.
            returns_type: Return type if specified.
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.returns_type = returns_type

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return (
            f"ParsedFunctionSchema(name={self.name!r}, "
            f"parameters={self.parameters!r})"
        )


def parse_function_definitions(
    definitions: List[FunctionDefinition],
) -> Dict[str, ParsedFunctionSchema]:
    """Translate Pydantic FunctionDefinition models into schema dictionary.

    Args:
        definitions: The list of validated FunctionDefinition models.

    Returns:
        A dictionary mapping each function name to its ParsedFunctionSchema.
    """
    schemas: Dict[str, ParsedFunctionSchema] = {}
    for fn in definitions:
        param_schemas: Dict[str, ParsedParameterSchema] = {}
        for param_name, param_model in fn.parameters.items():
            param_schemas[param_name] = ParsedParameterSchema(
                name=param_name,
                param_type=param_model.type,
                description=param_model.description,
                enum=param_model.enum,
                default=param_model.default,
            )
        returns_t = fn.returns.type if fn.returns else None
        schemas[fn.name] = ParsedFunctionSchema(
            name=fn.name,
            description=fn.description,
            parameters=param_schemas,
            returns_type=returns_t,
        )
    return schemas
