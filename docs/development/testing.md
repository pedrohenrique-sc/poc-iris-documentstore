---
title: Running Tests
---

# Running Tests

The test suite is split into two categories with distinct requirements.

---

## Test categories

| Category | Marker | IRIS required | Speed |
|---|---|---|---|
| Unit tests | `not integration` | No | Fast (~1s) |
| Integration tests | `integration` | Yes | Slow (~30s) |

---

## Running with hatch (recommended)

```bash
# Unit tests only — no IRIS needed, runs anywhere
hatch run test:unit

# Integration tests — IRIS must be running via docker-compose
hatch run test:integration

# All tests
hatch run test:all

# All tests with coverage
hatch run test:cov
```

---

## Running with pytest directly

If you prefer managing your own virtualenv:

```bash
pip install -e ".[dev]"

# Unit tests only
pytest tests/ -m "not integration" -v

# Integration tests
pytest tests/ -m "integration" -v

# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=haystack_integrations --cov-report=term-missing -v
```

---

## Environment variables for integration tests

```bash
export IRIS_CONNECTION_STRING="localhost:1972/USER"
export IRIS_USERNAME="_system"
export IRIS_PASSWORD="SYS"
```

Or create a `.env` file at the project root — `python-dotenv` is included in the test environment and loads it automatically if you call `load_dotenv()` in a conftest.

---

## Official Haystack mix-ins

Integration tests inherit from official Haystack test classes. These mix-ins are the same suite used to validate every official DocumentStore:

```python
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    WriteDocumentsTest,
)

@pytest.mark.integration
class TestCountDocuments(CountDocumentsTest):
    @pytest.fixture
    def document_store(self, document_store):
        return document_store
```

Passing all four mix-in classes is the minimum requirement for a Haystack-compatible DocumentStore.

---

## Writing new tests

### Unit test (no IRIS)

```python
class TestHelpers:
    def test_embedding_to_str(self):
        result = IRISDocumentStore._embedding_to_str([0.1, 0.2, 0.3])
        assert result == "[0.10000000,0.20000000,0.30000000]"

    def test_embedding_to_str_none(self):
        assert IRISDocumentStore._embedding_to_str(None) is None
```

### Integration test (requires IRIS)

```python
@pytest.mark.integration
class TestMyFeature:
    def test_something(self, document_store):
        document_store.write_documents([
            Document(id="t1", content="hello world")
        ])
        assert document_store.count_documents() == 1
```

### The `document_store` fixture

The shared fixture creates an isolated store using a dedicated `HaystackTest` table with `embedding_dim=4` (small vectors for fast insertion), cleans the table before and after each test, and closes the connection when done:

```python
@pytest.fixture()
def document_store():
    store = IRISDocumentStore(
        connection_string=Secret.from_token(os.getenv("IRIS_CONNECTION_STRING", "localhost:1972/USER")),
        username=Secret.from_token(os.getenv("IRIS_USERNAME", "_system")),
        password=Secret.from_token(os.getenv("IRIS_PASSWORD", "SYS")),
        table_name="HaystackTest",
        embedding_dim=4,
    )
    ids = [d.id for d in store.filter_documents()]
    store.delete_documents(ids)
    yield store
    ids = [d.id for d in store.filter_documents()]
    store.delete_documents(ids)
    store.close()
```

---

## CI configuration

Tests run automatically on every pull request and push to `main`. See `.github/workflows/tests.yml` for the full configuration. The CI matrix covers Python 3.10, 3.11, 3.12, and 3.13.

Integration tests run with IRIS as a Docker service in GitHub Actions:

```yaml
services:
  iris:
    image: intersystemsdc/iris-community:latest
    ports:
      - 1972:1972
    options: >-
      --health-cmd "iris status"
      --health-interval 15s
      --health-retries 5
      --health-start-period 45s
```