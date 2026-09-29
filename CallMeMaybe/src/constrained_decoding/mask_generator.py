"""Logits mask generator using NumPy for constrained decoding.

This module applies a constraint mask to the language model's output logits,
forcing all disallowed token IDs to negative infinity (-inf).
"""

from typing import Any, List, Sequence
import numpy as np


def apply_logit_mask(
    logits: Sequence[float] | np.ndarray[Any, Any],
    allowed_token_ids: List[int],
    mask_value: float = -np.inf,
) -> np.ndarray[Any, Any]:
    """Apply a constraint mask to logits such that only allowed token IDs remain selectable.

    Disallowed token positions are replaced with negative infinity (or mask_value),
    ensuring their probability becomes zero under softmax.

    Args:
        logits: Raw logit distribution over the vocabulary produced by the LLM.
        allowed_token_ids: List of integer token IDs permitted by the FSA.
        mask_value: Value used to penalize invalid tokens (defaults to -inf).

    Returns:
        A NumPy array of masked logits with the same shape and precision as the input.

    Raises:
        ValueError: If allowed_token_ids is empty, preventing unconstrained generation.
    """
    if not allowed_token_ids:
        raise ValueError("Cannot apply mask: allowed_token_ids list is empty.")

    masked_logits: np.ndarray[Any, Any] = np.full(len(logits), mask_value, dtype=np.float32)

    indices = np.array(allowed_token_ids, dtype=np.int64)
    valid_indices = indices[(indices >= 0) & (indices < len(logits))]

    original_array = np.asarray(logits, dtype=np.float32)
    masked_logits[valid_indices] = original_array[valid_indices]

    return masked_logits
