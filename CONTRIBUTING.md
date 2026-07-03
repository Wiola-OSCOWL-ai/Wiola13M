# Contributing to Wiola

Thanks for your interest in improving Wiola! Contributions of all kinds are
welcome — bug reports, docs, tests, new variants, kernels, and experiments.

## Development setup

```bash
git clone https://github.com/wiola-project/wiola.git
cd wiola
python -m venv .venv && source .venv/bin/activate   # Windows gitbash: source .venv/Scripts/activate
pip install -e ".[dev,train,hub]"
```

## Before opening a PR

```bash
ruff check src tests examples
black --check src tests examples
pytest
```

- Keep public APIs backward compatible where possible.
- New architectural options should be gated behind a `WiolaConfig` field with a
  sensible default and a test.
- Anything touching attention/caching must keep the
  `test_kv_cache_matches_full_forward` and `test_causality_no_future_leak` tests
  green — cached decoding must equal a full-sequence forward, and no future token
  may influence an earlier position.

## Commit / PR style

- Small, focused PRs with a clear description.
- Reference related issues.
- Add or update tests and docs alongside code.

## Code of conduct

Be respectful and constructive. We follow the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/).
