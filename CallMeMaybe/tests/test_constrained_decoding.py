"""Unit test suite for constrained decoding and schema validation."""

import unittest
import numpy as np
from src.constrained_decoding.mask_generator import apply_logit_mask
from src.constrained_decoding.json_schema_parser import (
    parse_function_definitions,
)
from src.models.functions import FunctionDefinition, ParameterSchema
from src.models.output_data import FunctionCallResult


class TestConstrainedDecoding(unittest.TestCase):
    """Test suite verifying logits masking, parsing, and models."""

    def test_apply_logit_mask_basic(self) -> None:
        """Verify that disallowed tokens are masked with -inf."""
        logits = [2.0, 5.0, 1.0, 8.0]
        allowed = [1, 3]
        masked = apply_logit_mask(logits, allowed)
        self.assertEqual(masked[1], 5.0)
        self.assertEqual(masked[3], 8.0)
        self.assertEqual(masked[0], -np.inf)
        self.assertEqual(masked[2], -np.inf)

    def test_apply_logit_mask_empty_raises(self) -> None:
        """Verify that empty allowed tokens list raises ValueError."""
        with self.assertRaises(ValueError):
            apply_logit_mask([1.0, 2.0], [])

    def test_parse_function_definitions(self) -> None:
        """Verify parsing Pydantic function models into schema dict."""
        func = FunctionDefinition(
            name="fn_test",
            description="A test function",
            parameters={
                "val": ParameterSchema(type="number", description="A number")
            },
        )
        schemas = parse_function_definitions([func])
        self.assertIn("fn_test", schemas)
        self.assertEqual(schemas["fn_test"].parameters["val"].type, "number")

    def test_function_call_result_valid(self) -> None:
        """Verify that valid output data passes Pydantic validation."""
        result = FunctionCallResult(
            prompt="test prompt",
            name="fn_test",
            parameters={"val": 42},
        )
        dumped = result.model_dump()
        self.assertEqual(dumped["name"], "fn_test")
        self.assertEqual(dumped["parameters"]["val"], 42)


if __name__ == "__main__":
    unittest.main()
