# Benchmarks

`context_transport.py` measures one narrow property: the estimated repository text carried
by a `load_deep_context` result compared with blindly sending every eligible source file.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/context_transport.py my-app \
  "Trace this cross-layer change and identify its regression tests"
```

This baseline is intentionally labelled weak. Coding agents normally use selective reads,
not whole-repository prompts. A large reduction here proves that the budget works; it does
not prove an equivalent reduction against a capable shell-using agent and says nothing by
itself about correctness.

Use [`docs/EVALUATION.md`](../docs/EVALUATION.md) for paired agent experiments. Publish raw
run artifacts, including unsuccessful runs, before making a percentage claim.
