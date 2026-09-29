"""Finite State Automaton (FSA) for JSON constrained decoding.

This module guides the token-by-token generation of structured JSON function calls,
ensuring strict compliance with the target schema and 100% syntactically valid JSON.
"""

from enum import Enum
import json
from typing import Any, Dict, List, Optional, Set
from src.constrained_decoding.json_schema_parser import ParsedFunctionSchema


class States(str, Enum):
    """Enumeration of possible states in the JSON function-calling FSA."""

    START = "START"
    NAME_KEY = "NAME_KEY"
    NAME_COLON = "NAME_COLON"
    FUNCTION_NAME = "FUNCTION_NAME"
    COMMA = "COMMA"
    PARAM_KEY = "PARAM_KEY"
    PARAM_COLON = "PARAM_COLON"
    PARAM_DICT_START = "PARAM_DICT_START"
    ARG_KEY = "ARG_KEY"
    ARG_COLON = "ARG_COLON"
    ARG_VALUE = "ARG_VALUE"
    END = "END"
    DONE = "DONE"


class FSAJSON:
    """Finite State Automaton governing valid token transitions for JSON generation."""

    def __init__(self, schemas: Dict[str, ParsedFunctionSchema], model: Any) -> None:
        """Initialize the FSA with function schemas and model encoder/decoder.

        Args:
            schemas: Mapping of function names to their parsed schemas.
            model: Instance of Small_LLM_Model providing encode() and decode().
        """
        self.schemas = schemas
        self.model = model
        self.state = States.START
        self.selected_function: Optional[str] = None
        self.buffer: str = ""
        self.full_text: str = ""

        # Tracking parameters for the selected function
        self.remaining_params: List[str] = []
        self.current_param: Optional[str] = None
        self.current_param_type: Optional[str] = None

    def is_finished(self) -> bool:
        """Check if the state machine has reached its terminal DONE state."""
        return self.state == States.DONE

    def get_valid_token_ids(self) -> List[int]:
        """Determine and return the list of allowed token IDs for the current state.

        Returns:
            A deduplicated list of integer token IDs.
        """
        allowed_strings = self._get_allowed_strings()
        token_ids: Set[int] = set()

        for s in allowed_strings:
            encoded = self.model.encode(s)
            ids = encoded.tolist()[0] if hasattr(encoded, "tolist") else list(encoded)
            for tid in ids:
                token_ids.add(int(tid))

        return list(token_ids)

    def _get_allowed_strings(self) -> List[str]:
        """Return the allowed string representations matching the current state."""
        match self.state:
            case States.START:
                return ["{"]
            case States.NAME_KEY:
                return ['"name"']
            case States.NAME_COLON | States.PARAM_COLON | States.ARG_COLON:
                return [":"]
            case States.FUNCTION_NAME:
                return [f'"{fn_name}"' for fn_name in self.schemas.keys()]
            case States.COMMA:
                return [","]
            case States.PARAM_KEY:
                return ['"parameters"']
            case States.PARAM_DICT_START:
                return ["{"]
            case States.ARG_KEY:
                if self.remaining_params:
                    return [f'"{self.remaining_params[0]}"']
                return []
            case States.ARG_VALUE:
                return self._get_allowed_value_strings()
            case States.END:
                return ["}"]
            case _:
                return []

    def _get_allowed_value_strings(self) -> List[str]:
        """Return allowed tokens depending on the expected parameter data type."""
        match self.current_param_type:
            case "number":
                delimiters = [","] if len(self.remaining_params) > 1 else ["}"]
                digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"]
                if any(c.isdigit() for c in self.buffer):
                    return digits + delimiters
                return digits
            case "boolean":
                return ["true", "false"]
            case "string":
                if not self.buffer.startswith('"'):
                    return ['"']
                return ['"']
            case _:
                return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", '""']

    def step(self, token_id: int) -> None:
        """Advance the automaton state based on the generated token ID.

        Args:
            token_id: The integer ID of the generated token.
        """
        token_str = self.model.decode([token_id])
        self.buffer += token_str
        self.full_text += token_str

        match self.state:
            case States.START if "{" in self.buffer:
                self.state, self.buffer = States.NAME_KEY, ""

            case States.NAME_KEY if '"name"' in self.buffer:
                self.state, self.buffer = States.NAME_COLON, ""

            case States.NAME_COLON if ":" in self.buffer:
                self.state, self.buffer = States.FUNCTION_NAME, ""

            case States.FUNCTION_NAME if self.buffer.count('"') >= 2:
                cleaned = self.buffer.strip().replace('"', "")
                if cleaned in self.schemas:
                    self.selected_function = cleaned
                    schema = self.schemas[cleaned]
                    self.remaining_params = list(schema.parameters.keys())
                    self.state, self.buffer = States.COMMA, ""

            case States.COMMA if "," in self.buffer:
                self.state, self.buffer = States.PARAM_KEY, ""

            case States.PARAM_KEY if '"parameters"' in self.buffer:
                self.state, self.buffer = States.PARAM_COLON, ""

            case States.PARAM_COLON if ":" in self.buffer:
                self.state, self.buffer = States.PARAM_DICT_START, ""

            case States.PARAM_DICT_START if "{" in self.buffer:
                if self.remaining_params:
                    self.current_param = self.remaining_params[0]
                    assert self.selected_function is not None
                    self.current_param_type = self.schemas[self.selected_function].parameters[self.current_param]
                    self.state, self.buffer = States.ARG_KEY, ""
                else:
                    self.state, self.buffer = States.END, ""

            case States.ARG_KEY if self.current_param and f'"{self.current_param}"' in self.buffer:
                self.state, self.buffer = States.ARG_COLON, ""

            case States.ARG_COLON if ":" in self.buffer:
                self.state, self.buffer = States.ARG_VALUE, ""

            case States.ARG_VALUE:
                self._handle_arg_value_transition()

            case States.END if "}" in self.buffer:
                self.state, self.buffer = States.DONE, ""

    def _handle_arg_value_transition(self) -> None:
        """Handle state progression when consuming parameter value tokens."""
        match self.current_param_type:
            case "number":
                if self.buffer.endswith(",") and len(self.remaining_params) > 1:
                    self.remaining_params.pop(0)
                    self.current_param = self.remaining_params[0]
                    assert self.selected_function is not None
                    self.current_param_type = self.schemas[self.selected_function].parameters[self.current_param]
                    self.state, self.buffer = States.ARG_KEY, ""
                elif self.buffer.endswith("}"):
                    self.remaining_params.clear()
                    self.state, self.buffer = States.END, ""

            case "boolean":
                if "true" in self.buffer or "false" in self.buffer:
                    if len(self.remaining_params) > 1:
                        self.remaining_params.pop(0)
                        self.current_param = self.remaining_params[0]
                        assert self.selected_function is not None
                        self.current_param_type = self.schemas[self.selected_function].parameters[self.current_param]
                        self.state, self.buffer = States.ARG_KEY, ""
                    else:
                        self.state, self.buffer = States.END, ""

            case "string":
                if self.buffer.count('"') >= 2:
                    if len(self.remaining_params) > 1:
                        self.remaining_params.pop(0)
                        self.current_param = self.remaining_params[0]
                        assert self.selected_function is not None
                        self.current_param_type = self.schemas[self.selected_function].parameters[self.current_param]
                        self.state, self.buffer = States.ARG_KEY, ""
                    else:
                        self.state, self.buffer = States.END, ""

    def get_parsed_json(self) -> Dict[str, Any]:
        """Parse and return the generated full text as a Python dictionary.

        Returns:
            The parsed dictionary representation of the generated JSON.
        """
        try:
            return json.loads(self.full_text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return {
                "name": self.selected_function or "",
                "parameters": {},
            }
