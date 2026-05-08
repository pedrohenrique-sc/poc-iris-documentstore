---
title: Credentials & Secrets
---

# Credentials & Secrets

`iris-haystack` uses Haystack's [`Secret`](https://docs.haystack.deepset.ai/reference/utils-api#secret)
class to handle IRIS credentials safely. Credentials are **never stored as plain strings**
and are **never serialized** in pipeline YAML files.

---

## The three credentials

IRIS requires three pieces of information to establish a connection:

| Credential | Environment variable | Description |
|---|---|---|
| Connection string | `IRIS_CONNECTION_STRING` | `host:port/namespace` |
| Username | `IRIS_USERNAME` | IRIS user account |
| Password | `IRIS_PASSWORD` | IRIS user password |

---

## Method 1 — Environment variables (recommended)

Set the variables in your shell before running your application:

```bash
export IRIS_CONNECTION_STRING="localhost:1972/USER"
export IRIS_USERNAME="_system"
export IRIS_PASSWORD="SYS"
```

Then create the store without any credential arguments — the defaults read from env vars:

```python
from haystack_integrations.document_stores.iris import IRISDocumentStore

store = IRISDocumentStore(embedding_dim=384)
# Credentials resolved from environment at connection time
```

This is the only approach that works correctly with pipeline YAML serialization.

---

## Method 2 — `.env` file

For local development, create a `.env` file at the project root:

```ini title=".env"
IRIS_CONNECTION_STRING=localhost:1972/USER
IRIS_USERNAME=_system
IRIS_PASSWORD=SYS
```

Load it before instantiating the store:

```python
from dotenv import load_dotenv
load_dotenv()

from haystack_integrations.document_stores.iris import IRISDocumentStore
store = IRISDocumentStore(embedding_dim=384)
```

!!! danger "Add `.env` to `.gitignore`"
    ```bash
    echo ".env" >> .gitignore
    git rm --cached .env  # remove if already tracked
    ```

---

## Method 3 — Inline `Secret.from_token` (testing only)

For unit tests or one-off scripts where env vars are inconvenient:

```python
from haystack.utils import Secret
from haystack_integrations.document_stores.iris import IRISDocumentStore

store = IRISDocumentStore(
    connection_string=Secret.from_token("localhost:1972/USER"),
    username=Secret.from_token("_system"),
    password=Secret.from_token("SYS"),
    embedding_dim=4,   # small for tests
)
```

!!! warning "Not for production"
    `Secret.from_token` keeps the value in memory as plain text. Any call to
    `store.to_dict()` will include the token value in the output — do not serialize
    pipelines that use inline tokens.

---

## How `Secret` protects credentials

### At connection time

```python
# Secret.from_env_var defers resolution — no env lookup at import time
conn = Secret.from_env_var("IRIS_CONNECTION_STRING")

# The value is only resolved when .resolve_value() is called
# (inside IRISDocumentStore._connect_with_retry)
conn_str = conn.resolve_value()  # reads $IRIS_CONNECTION_STRING now
```

### At serialization time

```python
store = IRISDocumentStore(embedding_dim=384)

d = store.to_dict()
print(d["init_parameters"]["password"])
# {'type': 'env_var', 'env_vars': ['IRIS_PASSWORD'], 'strict': True}
# ↑ env var NAME, not the resolved value
```

The YAML file is safe to commit because it contains only metadata about
where to find the credential, not the credential itself.

---

## Using custom environment variable names

If your deployment uses different env var names, pass the name to `from_env_var`:

```python
from haystack.utils import Secret

store = IRISDocumentStore(
    connection_string=Secret.from_env_var("PROD_IRIS_CONN"),
    username=Secret.from_env_var("PROD_IRIS_USER"),
    password=Secret.from_env_var("PROD_IRIS_PASS"),
    embedding_dim=384,
)
```

---

## Multiple IRIS environments

Use different env var sets for each environment:

=== "Development"
    ```bash
    export IRIS_CONNECTION_STRING="localhost:1972/USER"
    export IRIS_USERNAME="_system"
    export IRIS_PASSWORD="SYS"
    ```

=== "Staging"
    ```bash
    export IRIS_CONNECTION_STRING="staging-iris.internal:1972/STAGING"
    export IRIS_USERNAME="haystack_stage"
    export IRIS_PASSWORD="stage-password"
    ```

=== "Production"
    ```bash
    export IRIS_CONNECTION_STRING="iris-prod-01.internal:1972/PROD"
    export IRIS_USERNAME="haystack_prod"
    export IRIS_PASSWORD="prod-password"  # injected by secrets manager
    ```

The same `IRISDocumentStore()` call (no arguments) works in all environments —
the credentials come from the environment.

---

## Secrets managers

For production deployments, inject credentials via your secrets manager:

=== "AWS Secrets Manager"
    ```python
    import boto3, json
    secret = boto3.client("secretsmanager").get_secret_value(SecretId="iris/prod")
    creds = json.loads(secret["SecretString"])

    import os
    os.environ["IRIS_CONNECTION_STRING"] = creds["connection_string"]
    os.environ["IRIS_USERNAME"] = creds["username"]
    os.environ["IRIS_PASSWORD"] = creds["password"]

    from haystack_integrations.document_stores.iris import IRISDocumentStore
    store = IRISDocumentStore(embedding_dim=384)
    ```

=== "HashiCorp Vault"
    ```bash
    # Vault agent injects secrets as env vars automatically
    # or use envconsul / vault kv get
    vault kv get -field=password secret/iris/prod
    ```

=== "Kubernetes Secrets"
    ```yaml
    # Mount IRIS credentials as environment variables
    env:
      - name: IRIS_CONNECTION_STRING
        valueFrom:
          secretKeyRef:
            name: iris-credentials
            key: connection_string
      - name: IRIS_PASSWORD
        valueFrom:
          secretKeyRef:
            name: iris-credentials
            key: password
    ```