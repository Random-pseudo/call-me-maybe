"""Finite State Automaton (FSA) for JSON constrained decoding.

This module guides the token-by-token generation of structured JSON,
ensuring strict compliance with the target schema and 100% valid JSON.
"""

from enum import Enum
import json
from typing import Any, Dict, List, Optional
from src.constrained_decoding.json_schema_parser import (
    ParsedFunctionSchema,
    ParsedParameterSchema,
)


class States(str, Enum):
    """Enumeration of possible states in the JSON function-calling FSA."""

    START = "START"
    FN_KEY = "FN_KEY"
    FN_COLON = "FN_COLON"
    FN_VALUE = "FN_VALUE"
    COMMA_ARGS = "COMMA_ARGS"
    ARGS_KEY = "ARGS_KEY"
    ARGS_COLON = "ARGS_COLON"
    ARGS_START = "ARGS_START"
    PARAM_KEY = "PARAM_KEY"
    PARAM_COLON = "PARAM_COLON"
    PARAM_VALUE = "PARAM_VALUE"
    PARAM_COMMA = "PARAM_COMMA"
    ARGS_END = "ARGS_END"
    OBJ_END = "OBJ_END"
    DONE = "DONE"


class FSAJSON:
    """Automaton governing valid token transitions for JSON generation."""

    def __init__(
        self,
        schemas: Dict[str, ParsedFunctionSchema],
        model: Any,
    ) -> None:
        """Initialize the FSA with function schemas and model.

        Args:
            schemas: Mapping of function names to their parsed schemas.
            model: Instance of Small_LLM_Model providing encode/decode.
        """
        self.schemas = schemas
        self.model = model
        self.state = States.START
        self.selected_function: Optional[str] = None
        self.buffer: str = ""
        self.full_text: str = ""

        # Tracking parameters for the selected function
        self.remaining_params: List[str] = []
        self.current_param_name: Optional[str] = None
        self.current_param_schema: Optional[ParsedParameterSchema] = None
        self.parsed_args: Dict[str, Any] = {}

        # Cache standard token IDs using model.encode()
        self._newline_id = self._encode_first("\n")
        self._quote_id = self._encode_first('"')
        self._colon_id = self._encode_first(":")
        self._comma_id = self._encode_first(",")
        self._open_brace_id = self._encode_first("{")
        self._close_brace_id = self._encode_first("}")
        self._digit_ids = [self._encode_first(str(d)) for d in range(10)]
        self._minus_id = self._encode_first("-")
        self._dot_id = self._encode_first(".")

    def _encode_first(self, text: str) -> int:
        """Encode a single piece of text and return the first token ID."""
        encoded = self.model.encode(text)
        if hasattr(encoded, "tolist"):
            ids = encoded.tolist()[0]
        else:
            ids = list(encoded)
        return int(ids[0])

    def is_finished(self) -> bool:
        """Check if the state machine has reached terminal DONE state."""
        return self.state == States.DONE

    def get_valid_token_ids(self) -> List[int]:
        """Return the list of allowed token IDs for the current state.

        Returns:
            A deduplicated list of integer token IDs.
        """
        match self.state:
            case States.START:
                return [self._open_brace_id]

            case States.FN_KEY:
                # Target: '"fn_name"'
                return self._tokens_for_target('"fn_name"')

            case States.FN_COLON | States.ARGS_COLON | States.PARAM_COLON:
                return [self._colon_id]

            case States.FN_VALUE:
                # Allowed: prefixes of '"<fn_name>"' for available functions
                valid_ids: List[int] = []
                for fname in self.schemas.keys():
                    target = f'"{fname}"'
                    if target.startswith(self.buffer):
                        remaining = target[len(self.buffer):]
                        enc = self.model.encode(remaining)
                        if hasattr(enc, "tolist"):
                            ids = enc.tolist()[0]
                        else:
                            ids = list(enc)
                        if ids:
                            valid_ids.append(int(ids[0]))
                return list(set(valid_ids))

            case States.COMMA_ARGS | States.PARAM_COMMA:
                return [self._comma_id]

            case States.ARGS_KEY:
                return self._tokens_for_target('"args"')

            case States.ARGS_START:
                return [self._open_brace_id]

            case States.PARAM_KEY:
                if self.current_param_name:
                    return self._tokens_for_target(
                        f'"{self.current_param_name}"'
                    )
                return []

            case States.PARAM_VALUE:
                return self._get_allowed_param_value_tokens()

            case States.ARGS_END | States.OBJ_END:
                return [self._close_brace_id]

            case _:
                return []

    def _tokens_for_target(self, target: str) -> List[int]:
        """Return next valid token IDs to continue generating target string."""
        if target.startswith(self.buffer):
            remaining = target[len(self.buffer):]
            enc = self.model.encode(remaining)
            if hasattr(enc, "tolist"):
                ids = enc.tolist()[0]
            else:
                ids = list(enc)
            if ids:
                return [int(ids[0])]
        return []

    def _get_allowed_param_value_tokens(self) -> List[int]:
        """Return valid token IDs for current parameter value."""
        if not self.current_param_schema:
            return [self._quote_id]

        ptype = self.current_param_schema.type
        penum = self.current_param_schema.enum

        if penum:
            # Constrain to allowed enum options wrapped in quotes
            valid_ids: List[int] = []
            for opt in penum:
                target = f'"{opt}"'
                if target.startswith(self.buffer):
                    rem = target[len(self.buffer):]
                    enc = self.model.encode(rem)
                    if hasattr(enc, "tolist"):
                        ids = enc.tolist()[0]
                    else:
                        ids = list(enc)
                    if ids:
                        valid_ids.append(int(ids[0]))
            return list(set(valid_ids))

        if ptype in ["number", "integer", "float", "int"]:
            if len(self.remaining_params) <= 1:
                delimiters = [self._close_brace_id]
            else:
                delimiters = [self._comma_id]
            valid_numbers = list(self._digit_ids)
            if not self.buffer:
                valid_numbers.append(self._minus_id)
            if "." not in self.buffer and self.buffer:
                valid_numbers.append(self._dot_id)
            if self.buffer and any(c.isdigit() for c in self.buffer):
                valid_numbers.extend(delimiters)
            return list(set(valid_numbers))

        if ptype in ["boolean", "bool"]:
            valid_bools: List[int] = []
            for bstr in ["true", "false"]:
                if bstr.startswith(self.buffer):
                    rem = bstr[len(self.buffer):]
                    enc = self.model.encode(rem)
                    if hasattr(enc, "tolist"):
                        ids = enc.tolist()[0]
                    else:
                        ids = list(enc)
                    if ids:
                        valid_bools.append(int(ids[0]))
            return list(set(valid_bools))

        # Default string type: allow quote token
        return [self._quote_id]

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
                self.state, self.buffer = States.FN_KEY, ""

            case States.FN_KEY if '"fn_name"' in self.buffer:
                self.state, self.buffer = States.FN_COLON, ""

            case States.FN_COLON if ":" in self.buffer:
                self.state, self.buffer = States.FN_VALUE, ""

            case States.FN_VALUE if (
                self.buffer.startswith('"') and self.buffer.count('"') >= 2
            ):
                cleaned = self.buffer.strip().replace('"', "")
                if cleaned in self.schemas:
                    self.selected_function = cleaned
                    schema = self.schemas[cleaned]
                    self.remaining_params = list(schema.parameters.keys())
                    self.state, self.buffer = States.COMMA_ARGS, ""

            case States.COMMA_ARGS if "," in self.buffer:
                self.state, self.buffer = States.ARGS_KEY, ""

            case States.ARGS_KEY if '"args"' in self.buffer:
                self.state, self.buffer = States.ARGS_COLON, ""

            case States.ARGS_COLON if ":" in self.buffer:
                self.state, self.buffer = States.ARGS_START, ""

            case States.ARGS_START if "{" in self.buffer:
                if self.remaining_params:
                    self._prepare_next_param()
                    self.state, self.buffer = States.PARAM_KEY, ""
                else:
                    self.state, self.buffer = States.ARGS_END, ""

            case States.PARAM_KEY:
                if (
                    self.current_param_name
                    and f'"{self.current_param_name}"' in self.buffer
                ):
                    self.state, self.buffer = States.PARAM_COLON, ""

            case States.PARAM_COLON if ":" in self.buffer:
                self.state, self.buffer = States.PARAM_VALUE, ""

            case States.PARAM_VALUE:
                self._handle_param_value_transition()

            case States.PARAM_COMMA if "," in self.buffer:
                self._prepare_next_param()
                self.state, self.buffer = States.PARAM_KEY, ""

            case States.ARGS_END if "}" in self.buffer:
                self.state, self.buffer = States.OBJ_END, ""

            case States.OBJ_END if "}" in self.buffer:
                self.state, self.buffer = States.DONE, ""

    def _prepare_next_param(self) -> None:
        """Set up state pointers for the next parameter to generate."""
        if self.remaining_params and self.selected_function:
            self.current_param_name = self.remaining_params[0]
            func_schema = self.schemas[self.selected_function]
            self.current_param_schema = (
                func_schema.parameters[self.current_param_name]
            )

    def _handle_param_value_transition(self) -> None:
        """Handle state progression when consuming parameter value tokens."""
        if not self.current_param_schema or not self.current_param_name:
            return

        ptype = self.current_param_schema.type
        penum = self.current_param_schema.enum

        if penum:
            for opt in penum:
                if self.buffer == f'"{opt}"':
                    self.parsed_args[self.current_param_name] = opt
                    self._advance_after_param()
                    return

        elif ptype in ["number", "integer", "float", "int"]:
            if self.buffer.endswith(",") or self.buffer.endswith("}"):
                delim = self.buffer[-1]
                num_text = self.buffer[:-1].strip()
                if ptype in ["integer", "int"] and "." not in num_text:
                    self.parsed_args[self.current_param_name] = int(num_text)
                else:
                    self.parsed_args[self.current_param_name] = (
                        float(num_text) if "." in num_text else int(num_text)
                    )
                if delim == ",":
                    self.remaining_params.pop(0)
                    self._prepare_next_param()
                    self.state, self.buffer = States.PARAM_KEY, ""
                else:
                    self.remaining_params.clear()
                    self.state, self.buffer = States.ARGS_END, ""
                return

        elif ptype in ["boolean", "bool"]:
            if "true" in self.buffer or "false" in self.buffer:
                val = "true" in self.buffer
                self.parsed_args[self.current_param_name] = val
                self._advance_after_param()
                return

        else:
            # String type: ends when closing quote is reached
            if self.buffer.startswith('"') and self.buffer.count('"') >= 2:
                extracted = self.buffer[1:].split('"')[0]
                self.parsed_args[self.current_param_name] = extracted
                self._advance_after_param()
                return

    def _advance_after_param(self) -> None:
        """Advance to next parameter or end the args object."""
        self.remaining_params.pop(0)
        if self.remaining_params:
            self.state, self.buffer = States.PARAM_COMMA, ""
        else:
            self.state, self.buffer = States.ARGS_END, ""

    def get_parsed_json(self) -> Dict[str, Any]:
        """Parse and return generated full text as a dictionary.

        Returns:
            The parsed dictionary with exactly 'fn_name' and 'args' keys.
        """
        try:
            parsed = json.loads(self.full_text)
            if (
                isinstance(parsed, dict)
                and "fn_name" in parsed
                and "args" in parsed
            ):
                return {
                    "fn_name": str(parsed["fn_name"]),
                    "args": dict(parsed["args"]),
                }
        except (json.JSONDecodeError, Exception):
            pass

        return {
            "fn_name": self.selected_function or "",
            "args": self.parsed_args,
        }
