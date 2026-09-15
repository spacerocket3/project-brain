# Contributing

Project Brain is most useful when retrieval behavior is observable and reproducible.

## Before opening a change

- Describe the repository shape and the engineering question that exposed the problem.
- Separate retrieval evidence from conclusions about runtime behavior.
- Add a regression fixture for ranking, graph extraction, freshness, or context packing.
- Do not commit repositories, indexes, embeddings, credentials, or model caches.

## Development setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm ci
.venv/bin/python -m unittest -v
```

Run `git diff --check` before submitting. A retrieval improvement should explain what it
promotes, what it may demote, and how candidate and output bounds are preserved.

## Benchmark contributions

Agent benchmarks should use paired clean worktrees at one commit, identical tasks,
permissions, model settings, and time limits. Both agents must retain normal repository
tools. Report failures and retries as well as successful runs. See
[`docs/EVALUATION.md`](docs/EVALUATION.md).

By contributing, you agree that your contribution is licensed under Apache-2.0.
