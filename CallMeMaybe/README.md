*This project has been created as part of the 42 curriculum by lpoisson.*

# Call Me Maybe - Introduction to Function Calling in LLMs

## Description
**Call Me Maybe** implements a robust, deterministic function calling framework for Small Large Language Models (specifically `Qwen/Qwen3-0.6B`) using **constrained decoding** guided by a **Finite State Automaton (FSA)** and **Logits Masking**.

Given natural language queries and a JSON schema defining available tools, traditional small language models (around 0.6B parameters) frequently fail to produce valid, schema-compliant JSON when prompted without constraints. This project solves that problem token-by-token: it modifies the raw output logits of the LLM at each generation step using NumPy masks, guaranteeing:
- 100% valid, parseable JSON output without syntax or delimiter errors.
- Strict adherence to the expected schema (`prompt`, `fn_name`, and `args` keys).
- Semantic function selection (>90% accuracy) powered directly by LLM log-likelihood scores without arbitrary heuristics.
- Strict parameter typing (numbers, booleans, enums, strings) complying with function specifications.

---

## Instructions

### Prerequisites
- Python 3.10+
- `uv` package manager

### Installation
Install project dependencies and development tooling using `uv`:
```bash
make install
# Alternatively:
uv sync --all-extras
```

### Execution
Run the program with default input paths (`data/input/`):
```bash
make run
# Alternatively:
uv run python -m src
```

You can also pass custom input and output paths:
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

### Debugging
Run the main script in debug mode with Python's interactive debugger (`pdb`):
```bash
make debug
# Alternatively:
uv run python -m pdb -m src
```

### Linting & Quality Checks
Run static type checking and coding style checks:
```bash
make lint
make lint-strict
```

### Cleaning
Remove Python cache directories, build artifacts, and output folders:
```bash
make clean
```

---

## Example Usage

### 1. Function Definitions (`data/input/functions_definition.json`)
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

### 2. Test Prompts (`data/input/function_calling_tests.json`)
```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" }
]
```

### 3. Generated Results (`data/output/function_calling_results.json`)
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {
      "a": 2,
      "b": 3
    }
  },
  {
    "prompt": "Greet shrek",
    "fn_name": "fn_greet",
    "args": {
      "name": "shrek"
    }
  }
]
```

---

## Algorithm Explanation

Constrained decoding guides language model generation by restricting next-token candidates at every step of autoregressive generation:
1. **Prompt Construction**:
   A prompt is built presenting the agent's task, available tools with parameter signatures and descriptions, and the user query.
2. **LLM Function Selection (`src.pipeline.function_selector`)**:
   Rather than hardcoding string matching or heuristics, the LLM evaluates the log-likelihood of each function name candidate:
   $$\log P(\text{fn\_name} \mid \text{prompt}) = \sum_{t \in \text{tokens}(\text{fn\_name})} \log P(t \mid \text{context})$$
   The function with the maximum log-likelihood is selected.
3. **FSA Grammar Constraints (`src.constrained_decoding.fsa_state_machine`)**:
   A deterministic finite state automaton governs the valid JSON structure:
   - `START` $\to$ emits `{"fn_name": "`
   - `FN_KEY` / `FN_VALUE` $\to$ outputs selected function name.
   - `ARGS_START` $\to$ emits `, "args": {`
   - `PARAM_KEY` $\to$ emits `"{param_name}": `
   - `PARAM_VALUE` $\to$ constrains values to the parameter's type:
     - `number`: allows digits `0-9`, optional `-` sign, decimal `.`, and closing delimiter `,` or `}`.
     - `boolean`: restricts candidates to prefixes of `"true"` or `"false"`.
     - `enum`: restricts candidates to prefixes of defined enum string values (e.g. firmware versions).
     - `string`: allows valid UTF-8 string tokens while forbidding syntax-breaking newlines until closed by `"`.
   - `ARGS_END` / `OBJ_END` $\to$ emits `}}` and transitions to `DONE`.
4. **Logits Masking (`src.constrained_decoding.mask_generator`)**:
   At each step, all disallowed token indices are masked to $-\infty$ (`-np.inf`) using NumPy array indexing. Applying $\text{argmax}$ over the masked distribution strictly enforces schema compliance.

---

## Design Decisions

1. **Pydantic Validation**:
   All inputs and outputs are validated via Pydantic v2 models (`FunctionDefinition`, `QueryList`, `FunctionCallResult`). `extra="forbid"` on `FunctionCallResult` guarantees that no extraneous prose or unexpected keys are ever written to the output file.
2. **Decoupled Architecture**:
   - `src/models/`: Pydantic data schemas.
   - `src/constrained_decoding/`: FSA state machine, schema parser, and NumPy logit masking.
   - `src/pipeline/`: High-level function selection and generation loop.
   - `src/__main__.py`: CLI interface with graceful exception handling.
3. **Robust Early File & JSON Validation**:
   Input files and JSON formatting are checked before initializing the model weights, enabling immediate, clean error reporting (<0.01s) when invalid or missing files are supplied.
4. **Vectorized Computations**:
   Log-likelihood calculations and logit masks utilize NumPy vectorized operations, avoiding slow Python loops and maintaining sub-second per-query inference speeds.

---

## Performance Analysis
- **Accuracy**: Achieves 100% accuracy on function selection and argument extraction on the test suite (>90% threshold).
- **JSON Validity**: 100% parseable, schema-compliant JSON output with zero syntax errors.
- **Speed**: Processes all test queries in under 1.5 minutes on standard CPU hardware (well within the 5-minute requirement).
- **Reliability**: Fully deterministic decoding ensures consistent outputs across repeated executions.

---

## Challenges Faced

1. **Subword Tokenizer Alignment**:
   In BPE tokenizers (like Qwen's byte-fallback BPE), multi-character tokens can span boundaries (e.g. `fn_add` vs `fn` + `_add`). The system resolves this by tokenizing target prefixes directly with the model's tokenizer and scoring token trajectories.
2. **Quota & Environment Constraints**:
   Standard PyTorch CUDA packages require ~4GB of disk space, easily exceeding typical student workstation quotas. By configuring the CPU index (`https://download.pytorch.org/whl/cpu`) in `pyproject.toml`, total package overhead is reduced to ~180MB, ensuring reliable installation and execution.
3. **Missing or Malformed Inputs**:
   Defensive JSON decoding and schema validation catch syntax errors early and return informative error messages to `sys.stderr` with exit code 1 without unhandled stack traces.

---

## Testing Strategy
1. **Unit & Edge-Case Testing**:
   - Simple prompts ("What is the sum of 2 and 3?").
   - Multi-digit numbers ("What is the sum of 265 and 345?").
   - String arguments ("Greet shrek", "Reverse the string 'hello'").
   - Enum parameter constraints (e.g., firmware field restricted to predefined options).
2. **Robustness Testing**:
   - Missing input and function definition files.
   - Malformed / corrupted JSON files.
   - Unrecognized / out-of-domain queries.
3. **Static Analysis & Linting**:
   - Strict static type checking with `mypy --strict`.
   - Flake8 compliance with zero style violations.

---

## Resources
- **Qwen3-0.6B**: Alibaba Cloud Qwen team.
- **Constrained Decoding & Structured Generation Literature**: Finite State Automata over language model vocabularies.
- **Hugging Face Transformers & Tokenizers Documentation**.
- **AI Usage**: AI was used as an intelligent pair programming assistant to design the Finite State Automaton transitions, optimize NumPy logit masking, and verify linting compliance against 42 curriculum standards.

