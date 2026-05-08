---
title: Prerequisites
---

# Prerequisites

Before installing `iris-haystack`, make sure your environment meets the following requirements.

---

## Python

**Python 3.10 or higher is required.**

```bash
python --version
# Python 3.10.x | 3.11.x | 3.12.x | 3.13.x | 3.14.x
```

??? info "Why 3.10+?"
    The project uses `X | Y` union type syntax (PEP 604) and `match` statements which require Python 3.10. The `pyproject.toml` enforces `requires-python = ">=3.10"`.

---

## Docker

IRIS Community Edition is distributed as a Docker image. You need Docker Desktop or Docker Engine installed.

=== "macOS / Windows"
    Download [Docker Desktop](https://www.docker.com/products/docker-desktop/).

=== "Linux (Ubuntu/Debian)"
    ```bash
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl enable --now docker
    sudo usermod -aG docker $USER   # allow running without sudo (re-login required)
    ```

Verify the installation:

```bash
docker --version
# Docker version 27.x.x, build ...
```

---

## IRIS Connection Details

Once IRIS is running (see [Docker Setup](docker.md)), the default connection details for IRIS Community Edition are:

| Parameter | Default value |
|---|---|
| Host | `localhost` |
| Port (DB-API / superserver) | `1972` |
| Namespace | `USER` |
| Username | `_system` |
| Password | `SYS` |

!!! warning "Change the default password"
    In any environment beyond local development, change the default password immediately via the [Management Portal](http://localhost:52773/csp/sys/UtilHome.csp).

---

## Environment Variables

`iris-haystack` uses Haystack `Secret` objects to read credentials from the environment. Set these before running any code:

```bash
export IRIS_CONNECTION_STRING="localhost:1972/USER"
export IRIS_USERNAME="_system"
export IRIS_PASSWORD="SYS"
```

Or create a `.env` file at the root of your project:

```ini title=".env"
IRIS_CONNECTION_STRING=localhost:1972/USER
IRIS_USERNAME=_system
IRIS_PASSWORD=SYS
```

!!! danger "Never commit `.env` to git"
    Add `.env` to your `.gitignore`. Credentials committed to git are a security risk.

    ```bash
    echo ".env" >> .gitignore
    ```

---

## Optional: hatch (for contributors)

If you plan to contribute to the project or run the full test suite, you will also need [hatch](https://hatch.pypa.io/):

```bash
pip install hatch
```

See the [Development Setup](../development/hatch.md) guide for full details.