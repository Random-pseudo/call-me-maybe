*This project has been created as part of the 42 curriculum by eleve.*

# Call Me Maybe - Introduction to Function Calling in LLMs

## Description
This project implements a function calling mechanism for small Large Language Models (specifically Qwen/Qwen3-0.6B) using constrained decoding with a Finite State Automaton (FSA).
Given natural language queries and a JSON schema defining available tools, the system constrains the LLM token generation step-by-step to guarantee 100% valid JSON matching the exact schema.

## Instructions
### Installation
```bash
uv sync --all-extras
```

### Execution
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

### Linting
```bash
flake8 .
mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
```

## Resources
- Qwen3-0.6B by Qwen team (Alibaba Cloud)
- Constrained Decoding literature & Finite State Automata for structured grammar generation.
