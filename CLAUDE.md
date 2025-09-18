# Project
- Elelem is an "Inference provider" selector
- It defines providers and models you can use

# Provider definitions
- Providers (Groq, Scaleway, ...) are defined in src/elelem/providers/, one line per provider
- Models are defined in provider files
- Provider-wide flags can be passed in default_params (will be added as a header to all requests Elelem makes toward this provider)
- Example 
    provider:
    endpoint: {OPEN_AI_API ENDPOINT}
    default_params:
        param_1: param1_value
- Additional models can be passed at runtime through an ENV variable

# Model Syntax
Models are defined in every provider fles for convenience

models:
  "{MODEL_KEY}":
    metadata_ref: "{MODEL_METADATA}" # defined in src/elelem/providers/_metadata.yaml
    provider: {INFRASTRUCTURE_PROVIDER}
    model_id: {ID_AT_THE_PROVIDER} # How is the model referenced by the provider itself
    capabilities:
      supports_json_mode: {BOOLEAN} # Indicates if the model supports "response_format: json_object syntax. When set to false, Elelem will remove this flag BUT still expect JSON in output
      supports_temperature: {BOOLEAN} # Indicates if the model supports temperature adjustment
      supports_system: {BOOLEAN} # Indicates if system role messages are supported
    cost:
      input_cost_per_1m: {COST} Price per 1M input token 
      output_cost_per_1m:  {COST} Price per 1M output token (includes reasoning tokens)
      currency: USD

# Testing
- When making edits to the model definitions, launching tests/test_config_validation.py is recommended (uv run pytest tests/test_config_validation.py -v)
- When making edits to the core code, launching tests/test_elemem_with_faker.py is recommended (uv run pytest tests/test_elemem_with_faker.py -v)
- Before tagging or merging, or after large edits, launching the full pytest is of course recommended

# Model Faker
- A "Model Faker" is defined in tests/faker. This model is a fale model that answers to a pure OpenAI API, but its behavior can be full configured. It's meant to test Elelem thoroughly by emulating typical LLM problems.
- The model faker has "scenarios" in tests/faker/config/scenarios that can be used to emulate such failures or behaviors.

# Python Package Management with uv

Use uv exclusively for Python package management in this project.

## Package Management Commands

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`

## Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools like Pytest with `uv run pytest` or `uv run ruff`
- Launch a Python repl with `uv run python`

## Managing Scripts with PEP 723 Inline Metadata

- Run a Python script with inline metadata (dependencies defined at the top of the file) with: `uv run script.py`
- You can add or remove dependencies manually from the `dependencies =` section at the top of the script, or
- Or using uv CLI:
    - `uv add package-name --script script.py`
    - `uv remove package-name --script script.py`

# Git Commit Strategy - CRITICAL

**🚨 NEVER use `git add .` or `git add -A` - This leads to disasters!**

## Commit Best Practices

**DO:**
- `git add src/specific_file.py` - Add individual files explicitly
- `git add src/elelem/` - Add specific directories when needed
- `git status` - Always review what will be committed BEFORE committing
- `git diff --cached` - Review staged changes before committing

**NEVER DO:**
- `git add .` - Adds ALL files including temp files, test results, logs
- `git add -A` - Same disaster as above
- `git add *` - Shell glob expansion, unpredictable results
- `git commit -am` - Bypasses explicit file selection

## Safe Commit Workflow

1. `git status` - See what's changed
2. `git add src/elelem/specific_file.py` - Add only the files you modified
3. `git status` - Verify only intended files are staged
4. `git diff --cached` - Review the actual changes
5. `git commit -m "descriptive message"`

**When in doubt, add files one by one explicitly!**
