"""Logits mask generator using NumPy for constrained decoding.

This module applies a constraint mask to the language model's output logits,
forcing all disallowed token IDs to negative infinity (-inf).
"""

from typing import Any, List
import numpy as np


def apply_logit_mask(
    logits: List[float],
    allowed_token_ids: List[int],
    mask_value: float = -np.inf,
) -> np.ndarray[Any, Any]:
    """Apply a constraint mask to logits so only allowed IDs remain.

    Disallowed token positions are replaced with -inf (or mask_value),
    ensuring their probability becomes zero under softmax.

    Args:
        logits: Raw logit distribution over the vocabulary.
        allowed_token_ids: List of integer token IDs permitted by the FSA.
        mask_value: Value used to penalize invalid tokens (defaults to -inf).

    Returns:
        A NumPy array of masked logits with the same shape as the input.

    Raises:
        ValueError: If allowed_token_ids is empty.
    """
    if not allowed_token_ids:
        raise ValueError("Cannot apply mask: allowed_token_ids list is empty.")

    masked_logits = np.full(len(logits), mask_value)

    indices = np.array(allowed_token_ids)
    valid_indices = indices[(indices >= 0) & (indices < len(logits))]

    original_array = np.asarray(logits)
    masked_logits[valid_indices] = original_array[valid_indices]

    return masked_logits
