---
title: How to Contribute
---

# How to Contribute

All contributions are welcome — from fixing typos in docs to implementing new features.

---

## Types of contributions

| Type | Examples |
|---|---|
| Bug fix | A method returns wrong results, a connection error is not handled |
| New feature | SQL-based filtering, async support, HNSW auto-creation |
| Documentation | Fix a typo, add an example, expand a guide |
| Performance | BM25 index caching, batched INSERT |
| Tests | Cover an edge case, add a missing integration test |

---

## Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/iris-haystack.git
cd iris-haystack

# 3. Create a feature branch
git checkout -b feature/my-improvement

# 4. Start IRIS
docker-compose up -d

# 5. Verify everything works
hatch run test:all
```

---

## Development cycle

```bash
# Edit code ...

# Format and lint
hatch run fmt

# Run unit tests (fast)
hatch run test:unit

# Run full suite
hatch run test:all

# Check types
hatch run type-check
```

---

## Checklist before opening a PR

- [ ] All existing tests pass: `hatch run test:all`
- [ ] New functionality has tests
- [ ] New public methods have NumPy-style docstrings
- [ ] Code is formatted: `hatch run fmt-check` exits 0
- [ ] Type annotations are present on all public methods
- [ ] The `CHANGELOG.md` entry is added under `[Unreleased]`

---

## Commit message format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]
[optional footer]
```

Common types:

| Type | When to use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation change |
| `test` | Adding or fixing tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `chore` | Maintenance tasks (deps update, CI config) |

Examples:

```
feat(store): add SQL-level pre-filtering for large collections
fix(retriever): handle empty query_embedding gracefully
docs(hatch): add section on environment variables
test(bm25): add edge case for empty document collection
```

---

## Opening a Pull Request

1. Push your branch: `git push origin feature/my-improvement`
2. Open a PR on GitHub against the `main` branch
3. Fill in the PR template — describe what changed and why
4. Link any related issues with `Closes #N`
5. Wait for CI to pass — all checks must be green before merge

---

## Reporting bugs

Open an [issue on GitHub](https://github.com/s-c-ai/iris-haystack/issues) with:

- Python version
- `iris-haystack` version
- IRIS version (check `docker logs iris | grep "IRIS for"`)
- Minimal reproducible example
- Full traceback

---

## Requesting features

Open an issue with the `enhancement` label. Describe:

- The use case you are trying to solve
- The expected API (if applicable)
- Any constraints or trade-offs you are aware of