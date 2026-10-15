"""Main CLI entrypoint for Call Me Maybe.

Executes function calling pipeline using constrained decoding with Qwen3-0.6B.
"""

import argparse
import os
import sys


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the CLI.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling Constrained Decoding",
        add_help=True,
    )

    parser.add_argument(
        "--functions_definition",
        "--function_definitions",
        "--functions_definitions",
        "--function-definition",
        "--functions-definition",
        "-f",
        dest="functions_definition",
        default="data/input/functions_definition.json",
        help="Path to functions definition JSON file.",
    )

    parser.add_argument(
        "--input",
        "-i",
        "--input_file",
        "--input-file",
        dest="input_file",
        default="data/input/function_calling_tests.json",
        help="Path to input test prompts JSON file.",
    )

    parser.add_argument(
        "--output",
        "-o",
        "--output_file",
        "--output-file",
        dest="output_file",
        default="data/output/function_calling_results.json",
        help="Path to output JSON results file.",
    )

    return parser.parse_args()


def main() -> None:
    """Main execution function handling setup, validation, and pipeline."""
    args = parse_arguments()
    functions_path = args.functions_definition
    input_path = args.input_file
    output_path = args.output_file

    if not os.path.exists(functions_path):
        sys.stderr.write(
            f"Error: Function definitions file does not exist: "
            f"'{functions_path}'\n"
        )
        sys.exit(1)

    if not os.path.exists(input_path):
        sys.stderr.write(
            f"Error: Input tests file does not exist: '{input_path}'\n"
        )
        sys.exit(1)

    import json
    from pydantic import ValidationError
    from .functions import FunctionsDefinitionList
    from .input_data import QueryList

    try:
        with open(functions_path, "r", encoding="utf-8") as f:
            raw_functions = json.load(f)
    except Exception as e:
        sys.stderr.write(
            f"Error: Malformed JSON in function definitions file "
            f"'{functions_path}': {e}\n"
        )
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_inputs = json.load(f)
    except Exception as e:
        sys.stderr.write(
            f"Error: Malformed JSON in input tests file '{input_path}': {e}\n"
        )
        sys.exit(1)

    try:
        parsed_functions = FunctionsDefinitionList.model_validate(
            raw_functions
        )
    except ValidationError as e:
        sys.stderr.write(
            f"Error: Function definitions failed validation: {e}\n"
        )
        sys.exit(1)

    if not parsed_functions.root:
        sys.stderr.write(
            f"Error: No function definitions found in '{functions_path}'.\n"
        )
        sys.exit(1)

    try:
        QueryList.model_validate(raw_inputs)
    except ValidationError as e:
        sys.stderr.write(f"Error: Input tests failed validation: {e}\n")
        sys.exit(1)

    try:
        from llm_sdk.small_llm_model import Small_LLM_Model
        from .generator import run_pipeline

        model = Small_LLM_Model()
        run_pipeline(
            functions_path=functions_path,
            input_path=input_path,
            output_path=output_path,
            model=model,
        )

    except FileNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Unexpected error occurred: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
