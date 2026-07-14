# Contributing to evutils

Thanks for helping improve evutils. This guide covers the development setup and the
branching model. For release instructions, see [RELEASE.md](RELEASE.md).

## Development setup

evutils is a Python package with a C extension built by
[scikit-build-core](https://scikit-build-core.readthedocs.io/) (it fetches CMake
and Ninja itself; you only need a C compiler).

```bash
git clone --recurse-submodules https://github.com/mandulaj/evutils.git
cd evutils
# editable install with the test extras (uv recommended; plain pip works too)
uv pip install -e ".[test]"
```

Run the test suite:

```bash
python -m pytest tests/
```

Optional extras: `hdf5`, `aedat`, `docs` (e.g. `.[test,hdf5,aedat]`).

## Branching model

| Branch | Role |
|--------|------|
| `feature/*` | short-lived work; PR into `dev` (auto-deleted on merge) |
| `dev` | integration branch; every push publishes the `/dev/` docs |
| `main` | frozen, protected, linear history; all tags/releases are cut from here |

`main` requires **linear history** and **rebase-merge only** — so a branch must
be rebased onto the latest `main` before it merges.

## Making a change

1. Branch off `dev`: `git switch -c feature/my-change dev`.
2. Make the change; keep commits focused. Run `pytest` locally.
3. Open a PR into **`dev`**. CI runs the test matrix.
4. Rebase onto the latest `dev` if asked, then **rebase-merge**.
5. Every push to `dev` republishes the `/dev/` docs, so you can preview API/docs
   changes at `owner.github.io/evutils/dev/`.

## Documentation

Docs are Sphinx + AutoAPI on the pydata theme. Build locally:

```bash
uv pip install -e ".[docs]"
make -C docs html          # output in docs/build/html
```

They are published per-version to the `gh-pages` branch:

- `owner.github.io/evutils/` → latest **stable** (a copy of the newest final release)
- `owner.github.io/evutils/vX.Y.Z/` → a specific version
- `owner.github.io/evutils/dev/` → the `dev` branch
- `.../switcher.json` → the version-dropdown list

`docs/gen_switcher.py` rebuilds the switcher and mirrors stable to the root on
every deploy; `docs/deploy_versioned_docs.sh` pushes to `gh-pages` (plain git +
`GITHUB_TOKEN`, no third-party action).

## Version bumps

`pyproject.toml` `[project].version` is the **single source of truth**. The C
extension version (`EVUTILS_VERSION`) and the docs version derive from it
automatically — never edit them by hand. Helper:

```bash
python scripts/bump_version.py X.Y.Z
```

---

## Updating Python Versions

When adding support for a new Python version or dropping an old one, please ensure you update all hardcoded versions across the repository:

- [ ] `pyproject.toml`: Update `requires-python` and the `Programming Language :: Python :: 3.x` classifiers.
- [ ] `.github/workflows/test.yaml`: Update the `matrix.python` array.
- [ ] `.github/workflows/release.yaml`: Update the `matrix.python` array in the `smoke_pypi` job.
- [ ] `.github/workflows/release.yaml`: Update the `python-version: "3.x"` steps if you are changing the primary builder version.
