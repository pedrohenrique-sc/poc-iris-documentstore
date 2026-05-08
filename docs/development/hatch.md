---
title: Project Setup with hatch
---

# Project Setup with hatch

`iris-haystack` uses [hatch](https://hatch.pypa.io/) as its project manager. Hatch handles virtual environments, dependency installation, script execution, linting, and test execution — all without you having to manually activate virtualenvs or install `pytest` globally.

This page is the reference for every hatch command you need as a contributor.

---

## Why hatch?

| Task | Without hatch | With hatch |
|---|---|---|
| Create a virtualenv | `python -m venv .venv && source .venv/bin/activate` | automatic |
| Install dev deps | `pip install -r requirements-dev.txt` | automatic |
| Run tests | `pytest tests/` | `hatch run test:all` |
| Lint + format | `ruff check . && ruff format .` | `hatch run fmt` |
| Build wheel | `python -m build` | `hatch build` |
| Publish to PyPI | `twine upload dist/*` | `hatch publish` |

Hatch environments are **isolated and reproducible** — every contributor gets the same setup regardless of their local Python.

---

## Install hatch

=== "pip"
    ```bash
    pip install hatch
    ```
=== "pipx (recommended)"
    ```bash
    pipx install hatch
    ```
=== "brew (macOS)"
    ```bash
    brew install hatch
    ```

Verify:

```bash
hatch --version
# Hatch, version 1.x.x
```

---

## Project configuration overview

The full hatch configuration lives in `pyproject.toml`. Here is an annotated walkthrough of every relevant section.

### Build system

```toml title="pyproject.toml"
[build-system]
requires = ["hatchling"]        # (1)
build-backend = "hatchling.build"
```

1. `hatchling` is the build backend — it reads `[project]` metadata and creates the wheel/sdist. It is a required dependency only at build time, not at runtime.

### Version management

```toml
[tool.hatch.version]
path = "src/haystack_integrations/__about__.py"  # (1)
```

1. The version is read from the `__version__` variable in `__about__.py`. To change the version, edit that file. Do **not** set `version` in `[project]` directly — it is declared as `dynamic = ["version"]` instead.

```python title="src/haystack_integrations/__about__.py"
__version__ = "1.0.0"
```

### Package discovery

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/haystack_integrations"]  # (1)
```

1. Only the `haystack_integrations` namespace package is included in the wheel. Tests, examples, and docs are excluded from the published package.

---

## Environments

Hatch environments are isolated virtualenvs, each with its own dependencies and scripts.

### `default` — development tools

```toml title="pyproject.toml"
[tool.hatch.envs.default]
installer = "uv"           # (1)
dependencies = ["ruff"]    # (2)
```

1. Uses [uv](https://github.com/astral-sh/uv) as the installer inside hatch environments — significantly faster than pip.
2. Only `ruff` is installed here. Type checking (`mypy`) is run as a script but must be installed separately or added to the dependencies list.

**Available scripts:**

| Command | What it does |
|---|---|
| `hatch run fmt` | Fix linting issues + format code with ruff |
| `hatch run fmt-check` | Check formatting without making changes (used in CI) |
| `hatch run type-check` | Run mypy over `src/` and `tests/` |

```toml
[tool.hatch.envs.default.scripts]
fmt       = "ruff check --fix {args:.} && ruff format {args:.}"
fmt-check = "ruff check {args:.} && ruff format --check {args:.}"
type-check = "mypy src/haystack_integrations tests"
```

**Examples:**

```bash
# Format and fix all files
hatch run fmt

# Check without changing (CI mode)
hatch run fmt-check

# Run type checker
hatch run type-check

# Lint a specific file
hatch run fmt src/haystack_integrations/document_stores/iris/document_store.py
```

---

### `test` — test suite

```toml
[tool.hatch.envs.test]
dependencies = [
    "pytest",
    "pytest-cov",
    "pytest-asyncio",
    "python-dotenv",
]
```

!!! tip "Add `intersystems-irispython` to test env"
    The `dependencies` list in `[tool.hatch.envs.test]` does **not** inherit from `[project].dependencies` automatically in all hatch versions. If you get `ModuleNotFoundError: No module named 'iris'` in tests, add the core dependencies explicitly:

    ```toml
    [tool.hatch.envs.test]
    dependencies = [
        "haystack-ai",
        "intersystems-irispython",
        "pytest",
        "pytest-cov",
        "pytest-asyncio",
        "python-dotenv",
    ]
    ```

**Available scripts:**

| Command | What it does |
|---|---|
| `hatch run test:unit` | Only tests **not** marked `integration` — no IRIS needed |
| `hatch run test:integration` | Only tests marked `integration` — IRIS must be running |
| `hatch run test:all` | Every test |
| `hatch run test:cov` | Every test + coverage report |

```toml
[tool.hatch.envs.test.scripts]
unit        = 'pytest -m "not integration" {args:tests}'
integration = 'pytest -m "integration" {args:tests}'
all         = "pytest {args:tests}"
cov         = "pytest --cov=haystack_integrations {args:tests}"
```

**Examples:**

```bash
# Run unit tests only (fast, no IRIS required)
hatch run test:unit

# Run integration tests (IRIS must be running via docker-compose up -d)
hatch run test:integration

# Run everything and print coverage
hatch run test:cov

# Pass extra pytest flags
hatch run test:all -- -v -k "bm25"

# Run a single test file
hatch run test:unit -- tests/test_document_store.py

# Run a single test by name
hatch run test:unit -- -k "test_duplicate_policy_fail"
```

---

### `example` — run the pipeline example

```toml
[tool.hatch.envs.example]
dependencies = [
    "sentence-transformers",
    "python-dotenv",
]

[tool.hatch.envs.example.scripts]
run = "python examples/rag_pipeline.py"
```

```bash
# Run the complete RAG example (IRIS must be running)
hatch run example:run
```

---

## Pytest configuration

```toml title="pyproject.toml"
[tool.pytest.ini_options]
minversion = "6.0"
markers = [
    "unit: unit tests",
    "integration: integration tests",
]
addopts = ["--import-mode=importlib"]  # (1)
```

1. `--import-mode=importlib` ensures tests are discovered correctly even without `__init__.py` files in the `tests/` directory.

### Marking tests

```python title="tests/test_document_store.py"
import pytest

class TestCountDocuments:
    # No marker = runs in both unit and integration
    def test_embedding_to_str(self):
        ...

@pytest.mark.integration
class TestWriteDocuments:
    # Requires IRIS — only runs with hatch run test:integration
    def test_write_basic(self, document_store):
        ...
```

---

## Ruff configuration

Ruff is the linter and formatter. The configuration selects a comprehensive set of rules:

```toml title="pyproject.toml"
[tool.ruff]
line-length = 120

[tool.ruff.lint]
select = [
    "A",      # flake8-builtins
    "ANN",    # flake8-annotations (type hints)
    "ARG",    # flake8-unused-arguments
    "B",      # flake8-bugbear
    "C",      # mccabe complexity
    "D102",   # missing docstring in public method
    "D103",   # missing docstring in public function
    "D205",   # 1 blank line between summary and description
    "D209",   # closing triple quotes on new line
    "D213",   # summary on second physical line
    "D417",   # missing argument descriptions
    "D419",   # empty docstring
    "DTZ",    # datetime timezone
    "E",      # pycodestyle errors
    "EM",     # flake8-errmsg
    "F",      # pyflakes
    "I",      # isort
    "N",      # pep8-naming
    ...
]
ignore = [
    "B027",    # empty method in abstract base class — OK for protocols
    "S105",    # possible hardcoded password — false positives in tests
    "C901",    # too complex — handled case by case
    "PLR0913", # too many arguments — IRISDocumentStore.__init__ has many params
]
```

### Per-file overrides

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*"   = ["D", "PLR2004", "S101", "TID252", "ANN"]
"examples/**/*" = ["D", "T201", "ANN"]
```

- **Tests** are exempt from docstring rules (`D`), magic number checks (`PLR2004`), `assert` warnings (`S101`), and type annotation requirements (`ANN`).
- **Examples** are exempt from docstrings and `print()` warnings (`T201`).

---

## mypy configuration

```toml
[tool.mypy]
install_types       = true     # auto-install missing stubs
non_interactive     = true     # don't ask for confirmation
check_untyped_defs  = true     # check bodies of untyped functions
disallow_incomplete_defs = true

[[tool.mypy.overrides]]
module = ["haystack.*"]
ignore_missing_imports = true  # haystack ships its own stubs
```

---

## Coverage configuration

```toml
[tool.coverage.run]
source   = ["haystack_integrations"]
branch   = true    # measure branch coverage (if/else paths)
parallel = false

[tool.coverage.report]
omit = ["*/tests/*", "*/__init__.py"]
show_missing = true
exclude_lines = [
    "no cov",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

Generate an HTML coverage report:

```bash
hatch run test:cov -- --cov-report=html
open htmlcov/index.html
```

---

## Common workflows

### First-time setup

```bash
git clone https://github.com/s-c-ai/iris-haystack.git
cd iris-haystack

# Start IRIS
docker-compose up -d

# Run all tests (hatch creates the env automatically)
hatch run test:all
```

### Daily development loop

```bash
# Edit code...

# Check style
hatch run fmt-check

# Fix style issues automatically
hatch run fmt

# Run fast unit tests
hatch run test:unit

# Run full suite before committing
hatch run test:all
```

### Before opening a Pull Request

```bash
# 1. Format and lint
hatch run fmt

# 2. Type check
hatch run type-check

# 3. Full test suite with coverage
hatch run test:cov

# 4. Verify the package builds cleanly
hatch build
```

### Building and publishing

```bash
# Build wheel + sdist
hatch build
# Output: dist/intersystems_iris_haystack-1.0.0-py3-none-any.whl
#         dist/intersystems_iris_haystack-1.0.0.tar.gz

# Publish to PyPI (requires credentials or Trusted Publishing)
hatch publish
```

!!! warning "Publish to TestPyPI first"
    Always test on TestPyPI before the real index:
    ```bash
    hatch publish --repo test
    ```
    Then install from TestPyPI to verify:
    ```bash
    pip install -i https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/ \
                intersystems-iris-haystack
    ```