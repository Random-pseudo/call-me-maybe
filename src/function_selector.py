"""Function selector module using constrained decoding.

This module selects the appropriate function to call based on the LLM's
next-token logits while restricting output to the available function names.
"""

from typing import Any, Dict, List
import numpy as np
from .functions import FunctionDefinition
from .mask_generator import apply_logit_mask


def format_functions_doc(schemas: Dict[str, FunctionDefinition]) -> str:
    """Format available functions into a markdown string for system prompt.

    Args:
        schemas: Dictionary of parsed function schemas.

    Returns:
        Formatted multi-line string describing all available functions.
    """
    lines: List[str] = []
    for fn in schemas.values():
        param_parts: List[str] = []
        for name, parameter in fn.parameters.items():
            param_parts.append(f"{name}: {parameter.type}")
        params_str = ", ".join(param_parts)
        lines.append(f"- {fn.name}({params_str}): {fn.description}")
    return "\n".join(lines)


def _encode_ids(model: Any, text: str) -> List[int]:
    """Encode text and normalize the result to a flat list of token IDs."""
    encoded = model.encode(text)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def select_function_with_constrained_decoding(
    model: Any,
    query: str,
    schemas: Dict[str, FunctionDefinition],
    system_prompt: str,
) -> str:
    """Select a function by decoding only valid function-name tokens.

    Args:
        model: Instance of Small_LLM_Model providing inference methods.
        query: User natural language prompt.
        schemas: Mapping of function names to their parsed schemas.
        system_prompt: System prompt describing the agent's task and tools.

    Returns:
        A function name selected from ``schemas``.

    Raises:
        ValueError: If no function definitions are available, or if tokenizing
            a candidate changes the already-encoded prompt prefix.
        RuntimeError: If constrained decoding cannot match a function name.
    """
    if not schemas:
        raise ValueError("No function definitions available for selection.")

    if len(schemas) == 1:
        return next(iter(schemas.keys()))

    prefix = (
        f'{system_prompt}\nUser query: {query}\n'
        'JSON call: {"name": "'
    )
    prefix_ids = _encode_ids(model, prefix)
    candidates: Dict[str, List[int]] = {}

    for function_name in schemas:
        candidate_ids = _encode_ids(model, f'{prefix}{function_name}"')
        if candidate_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "Tokenizer changed the encoded prompt when appending "
                f"function name {function_name!r}."
            )
        name_ids = candidate_ids[len(prefix_ids):]
        if not name_ids:
            raise ValueError(
                f"Function name {function_name!r} produced no tokens."
            )
        candidates[function_name] = name_ids

    generated_ids: List[int] = []
    current_ids = list(prefix_ids)
    max_length = max(len(token_ids) for token_ids in candidates.values())

    for _ in range(max_length + 1):
        active = {
            name: token_ids
            for name, token_ids in candidates.items()
            if token_ids[: len(generated_ids)] == generated_ids
        }
        completed = [
            name
            for name, token_ids in active.items()
            if len(token_ids) == len(generated_ids)
        ]
        if completed:
            return completed[0]

        allowed_ids = list(
            {
                token_ids[len(generated_ids)]
                for token_ids in active.values()
                if len(token_ids) > len(generated_ids)
            }
        )
        if not allowed_ids:
            raise RuntimeError(
                "Constrained decoding found no valid next token for "
                "available function names."
            )

        logits = model.get_logits_from_input_ids(current_ids)
        next_id = int(np.argmax(apply_logit_mask(logits, allowed_ids)))
        generated_ids.append(next_id)
        current_ids.append(next_id)

    raise RuntimeError(
        "Constrained decoding did not finish a valid function name."
    )
