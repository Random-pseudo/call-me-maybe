"""Generator module orchestrating constrained decoding and function execution.

This module applies logits masking and token-by-token sampling to generate
strictly schema-compliant JSON function calls.
"""

import json
import os
import sys
from typing import Any, Dict, List
from tqdm import tqdm
import numpy as np
from .functions import (
    FunctionDefinition,
    FunctionsDefinitionList,
    ParameterSchema,
)
from .function_selector import (
    format_functions_doc,
    select_function_with_constrained_decoding,
)
from .input_data import QueryList
from .mask_generator import apply_logit_mask
from .output_data import FunctionCallResult


def _generate_integer(
    model: Any,
    curr_ids: List[int],
    is_last: bool,
) -> tuple[int, List[int]]:
    """Generate an integer parameter without allowing decimal points.

    Args:
        model: Instance of Small_LLM_Model.
        curr_ids: Current sequence of token IDs.
        is_last: Whether this is the final parameter in the function.

    Returns:
        A tuple of (parsed_int, updated_token_ids).
    """
    digit_ids = [int(model.encode(str(d))[0].tolist()[0]) for d in range(10)]
    minus_id = int(model.encode("-")[0].tolist()[0])
    delim_id = int(model.encode("}" if is_last else ",")[0].tolist()[0])

    num_str = ""
    for _ in range(20):
        logits = model.get_logits_from_input_ids(curr_ids)
        valid = list(digit_ids)
        if not num_str:
            valid.append(minus_id)
        if num_str and any(c.isdigit() for c in num_str):
            valid.append(delim_id)

        masked = apply_logit_mask(logits, valid)
        next_id = int(np.argmax(masked))
        if next_id == delim_id:
            break
        num_str += model.decode([next_id])
        curr_ids.append(next_id)

    clean = num_str.strip()
    val = int(clean) if clean.lstrip("-").isdigit() else 0
    return val, curr_ids


def _generate_float(
    model: Any,
    curr_ids: List[int],
    is_last: bool,
) -> tuple[float, List[int]]:
    """Generate a floating-point number parameter.

    Args:
        model: Instance of Small_LLM_Model.
        curr_ids: Current sequence of token IDs.
        is_last: Whether this is the final parameter in the function.

    Returns:
        A tuple of (parsed_float, updated_token_ids).
    """
    digit_ids = [int(model.encode(str(d))[0].tolist()[0]) for d in range(10)]
    minus_id = int(model.encode("-")[0].tolist()[0])
    dot_id = int(model.encode(".")[0].tolist()[0])
    delim_id = int(model.encode("}" if is_last else ",")[0].tolist()[0])

    num_str = ""
    for _ in range(20):
        logits = model.get_logits_from_input_ids(curr_ids)
        valid = list(digit_ids)
        if not num_str:
            valid.append(minus_id)
        if "." not in num_str and num_str:
            valid.append(dot_id)
        if num_str and any(c.isdigit() for c in num_str):
            valid.append(delim_id)

        masked = apply_logit_mask(logits, valid)
        next_id = int(np.argmax(masked))
        if next_id == delim_id:
            break
        num_str += model.decode([next_id])
        curr_ids.append(next_id)

    clean = num_str.strip()
    try:
        val = float(clean)
    except ValueError:
        val = 0.0
    return val, curr_ids


def _generate_bool(
    model: Any,
    curr_ids: List[int],
) -> tuple[bool, List[int]]:
    """Generate a boolean parameter ('true' or 'false').

    Args:
        model: Instance of Small_LLM_Model.
        curr_ids: Current sequence of token IDs.

    Returns:
        A tuple of (parsed_bool, updated_token_ids).
    """
    t_ids = [int(x) for x in model.encode("true")[0].tolist()]
    f_ids = [int(x) for x in model.encode("false")[0].tolist()]
    logits = model.get_logits_from_input_ids(curr_ids)
    chosen = bool(logits[t_ids[0]] >= logits[f_ids[0]])
    chosen_ids = t_ids if chosen else f_ids
    curr_ids.extend(chosen_ids)
    return chosen, curr_ids


def _generate_string(
    model: Any,
    curr_ids: List[int],
) -> tuple[str, List[int]]:
    """Generate a string parameter bounded by quotes.

    Args:
        model: Instance of Small_LLM_Model.
        curr_ids: Current sequence of token IDs.

    Returns:
        A tuple of (parsed_str, updated_token_ids).
    """
    quote_id = int(model.encode('"')[0].tolist()[0])
    newline_id = int(model.encode("\n")[0].tolist()[0])

    curr_ids.append(quote_id)
    val = ""
    for _ in range(30):
        logits = model.get_logits_from_input_ids(curr_ids)
        logits[newline_id] = -1e9
        next_id = int(np.argmax(logits))
        tok = model.decode([next_id])
        if '"' in tok:
            val += tok.split('"')[0]
            curr_ids.append(quote_id)
            break
        val += tok
        curr_ids.append(next_id)

    return val, curr_ids


def generate_parameter_value(
    model: Any,
    input_ids: List[int],
    param_schema: ParameterSchema,
    is_last: bool,
) -> tuple[Any, List[int]]:
    """Generate a single typed parameter value using constrained decoding.

    Args:
        model: Instance of Small_LLM_Model.
        input_ids: Current sequence of token IDs.
        param_schema: Parsed schema for the parameter.
        is_last: Whether this is the final parameter in the function.

    Returns:
        A tuple of (parsed_value, updated_token_ids).
    """
    curr_ids = list(input_ids)
    ptype = param_schema.type

    if ptype in ("integer", "int"):
        return _generate_integer(model, curr_ids, is_last)

    if ptype in ("number", "float"):
        return _generate_float(model, curr_ids, is_last)

    if ptype in ("boolean", "bool"):
        return _generate_bool(model, curr_ids)

    return _generate_string(model, curr_ids)


def generate_function_call(
    model: Any,
    query: str,
    schemas: Dict[str, FunctionDefinition],
    system_prompt: str,
) -> FunctionCallResult:
    """Generate schema-constrained function call for a single prompt.

    Args:
        model: Instance of Small_LLM_Model.
        query: Natural language prompt string.
        schemas: Dictionary of parsed function schemas.
        system_prompt: Base system instructions.

    Returns:
        A validated FunctionCallResult model with prompt, fn_name, and args.
    """
    selected_fn = select_function_with_constrained_decoding(
        model, query, schemas, system_prompt
    )
    schema = schemas[selected_fn]

    prompt_str = (
        f"{system_prompt}\nUser query: {query}\n"
        f'JSON call: {{"name": "{selected_fn}", "parameters": {{'
    )
    encoded_prompt = model.encode(prompt_str)
    if hasattr(encoded_prompt, "tolist"):
        curr_ids = encoded_prompt.tolist()[0]
    else:
        curr_ids = list(encoded_prompt)

    args_dict: Dict[str, Any] = {}
    params = list(schema.parameters.items())

    for i, (pname, pschema) in enumerate(params):
        is_last = i == len(params) - 1
        key_str = f'"{pname}": '
        curr_ids.extend([int(x) for x in model.encode(key_str)[0].tolist()])

        val, curr_ids = generate_parameter_value(
            model, curr_ids, pschema, is_last
        )
        args_dict[pname] = val

        if not is_last:
            curr_ids.extend([int(x) for x in model.encode(", ")[0].tolist()])

    curr_ids.extend([int(x) for x in model.encode("}}")[0].tolist()])

    return FunctionCallResult(
        prompt=query,
        name=selected_fn,
        parameters=args_dict,
    )


def run_pipeline(
    functions_path: str,
    input_path: str,
    output_path: str,
    model: Any,
) -> None:
    """Execute the entire function calling pipeline from input to output.

    Args:
        functions_path: Filepath to the function definitions JSON.
        input_path: Filepath to the test prompts JSON.
        output_path: Destination filepath for the results JSON.
        model: Instance of Small_LLM_Model.

    Raises:
        FileNotFoundError: If input files do not exist.
        ValueError: If input files contain invalid JSON.
    """
    if not os.path.exists(functions_path):
        raise FileNotFoundError(
            f"Function definitions file not found: {functions_path}"
        )
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input tests file not found: {input_path}")

    try:
        with open(functions_path, "r", encoding="utf-8") as f:
            raw_functions = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Malformed JSON in functions file '{functions_path}': {e}"
        ) from e

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_inputs = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Malformed JSON in input tests file '{input_path}': {e}"
        ) from e

    validated_functions = FunctionsDefinitionList.model_validate(
        raw_functions
    ).root
    validated_inputs = QueryList.model_validate(raw_inputs).root
    schemas = {function.name: function for function in validated_functions}

    funcs_doc = format_functions_doc(schemas)
    system_prompt = (
        "You are an expert function calling assistant. "
        "Given the user query and available functions, "
        "output a JSON object with 'name' and 'parameters'.\n\n"
        f"Available functions:\n{funcs_doc}\n"
    )

    results: List[Dict[str, Any]] = []
    for item in tqdm(validated_inputs, desc="Processing queries"):
        result_item = generate_function_call(
            model, item.prompt, schemas, system_prompt
        )
        results.append(result_item.model_dump())

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    sys.stdout.write(
        f"Successfully processed {len(results)} queries. "
        f"Output saved to {output_path}\n"
    )
