"""Function selector module using LLM probabilities.

This module selects the appropriate function to call based on the LLM's
semantic understanding of the natural language user prompt, without heuristics.
"""

from typing import Any, Dict, List
import numpy as np
from src.constrained_decoding.json_schema_parser import ParsedFunctionSchema


def format_functions_doc(schemas: Dict[str, ParsedFunctionSchema]) -> str:
    """Format available functions into a markdown string for system prompt.

    Args:
        schemas: Dictionary of parsed function schemas.

    Returns:
        Formatted multi-line string describing all available functions.
    """
    lines: List[str] = []
    for fn in schemas.values():
        param_parts: List[str] = []
        for p in fn.parameters.values():
            param_parts.append(f"{p.name}: {p.type}")
        params_str = ", ".join(param_parts)
        lines.append(f"- {fn.name}({params_str}): {fn.description}")
    return "\n".join(lines)


def select_function_with_llm(
    model: Any,
    query: str,
    schemas: Dict[str, ParsedFunctionSchema],
    system_prompt: str,
) -> str:
    """Select the best matching function name using LLM probabilities.

    Args:
        model: Instance of Small_LLM_Model providing inference methods.
        query: User natural language prompt.
        schemas: Mapping of function names to their parsed schemas.
        system_prompt: System prompt describing the agent's task and tools.

    Returns:
        The function name with highest log-likelihood score under the model.
    """
    if not schemas:
        raise ValueError("No function definitions available for selection.")

    if len(schemas) == 1:
        return next(iter(schemas.keys()))

    prefix = f"{system_prompt}\nUser query: {query}\n"
    base_prompt = f'{prefix}JSON call: {{"name": "'
    encoded_base = model.encode(base_prompt)
    if hasattr(encoded_base, "tolist"):
        base_ids = encoded_base.tolist()[0]
    else:
        base_ids = list(encoded_base)

    scores: Dict[str, float] = {}

    for fname in schemas.keys():
        target_str = f'{base_prompt}{fname}"'
        encoded_target = model.encode(target_str)
        if hasattr(encoded_target, "tolist"):
            target_ids = encoded_target.tolist()[0]
        else:
            target_ids = list(encoded_target)
        fname_ids = target_ids[len(base_ids):]

        score = 0.0
        curr_ids = list(base_ids)
        for tid in fname_ids:
            logits = model.get_logits_from_input_ids(curr_ids)
            arr = np.asarray(logits)
            max_l = float(np.max(arr))
            lse = max_l + float(np.log(np.sum(np.exp(arr - max_l))))
            score += float(arr[tid] - lse)
            curr_ids.append(tid)

        scores[fname] = score

    return max(scores, key=scores.get)  # type: ignore[arg-type]
