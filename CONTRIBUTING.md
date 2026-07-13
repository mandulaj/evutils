# Contributing to evutils

Thanks for helping improve evutils. This guide covers the development setup, the
branching model, and how releases are cut.

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

## Releasing

A release is **staged to TestPyPI first** and only promoted to real PyPI once it
checks out. `main` stays frozen and linear; tags are always cut from it.

```
                          ┌─────────────────┐
   feature/x  ──PR──▶     │      dev        │   push to dev
   (rebase-merge)         │  (integration)  │──────────────▶ dev-docs.yaml
                          └────────┬────────┘                 tests → publish /dev/
                                   │
                    ready to release vX.Y.Z
                                   │
              1. bump pyproject.toml → X.Y.Z (commit on dev)
                                   │
                        2. PR  dev ──▶ main   (rebase-merge, linear)
                                   │
                          ┌────────▼────────┐
                          │      main       │   (frozen, protected, linear)
                          └────────┬────────┘
                                   │
        3. GitHub Release: target=main, tag=vX.Y.Z, ☑ PRE-RELEASE
                                   │
                                   ▼
                        release.yaml  (prereleased)
                     check_version → tests → wheels+sdist
                                   │
                                   ▼
                        ▶ TestPyPI  +  smoke-install test
                          (NO real PyPI, NO docs)
                                   │
                 4. validate:  pip install -i test.pypi.org … evutils==X.Y.Z
                                   │
                        looks good?  ── no ──▶ fix, bump to next version, redo
                                   │ yes
        5. Edit the SAME release → untick pre-release → ☑ Set as latest
                                   │
                                   ▼
                        release.yaml  (released)
                     check_version → tests → wheels+sdist
                                   │
                   ┌───────────────┼───────────────┐
                   ▼               ▼               ▼
             ▶ real PyPI     full smoke      ▶ docs /vX.Y.Z/
             (X.Y.Z)         matrix (5×5)      + root mirror (new stable)
                                   │
              6. sync dev:  rebase/FF dev onto main (carry the bump)
```

### Steps

| # | Where | Action | Result |
|---|-------|--------|--------|
| 1 | `dev` | Bump `pyproject.toml` → `X.Y.Z`, commit | single source of truth set |
| 2 | PR `dev→main` | rebase-merge | tooling + version on `main` |
| 3 | GitHub UI | Release: target `main`, tag `vX.Y.Z`, **pre-release** | → **TestPyPI** + smoke; no PyPI, no docs |
| 4 | local | install from TestPyPI, verify | confidence before going public |
| 5 | GitHub UI | edit release → **full / latest** | → **real PyPI** + docs `/vX.Y.Z/` + root, full smoke matrix |
| 6 | `dev` | rebase/FF onto `main` | dev keeps the bump; no drift |

Validate from TestPyPI (step 4):

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            "evutils==X.Y.Z"
```

### Guardrails (in the workflows)

- **`check_version`** fails the release fast if the git tag ≠ `pyproject` version.
- **Pre-release → TestPyPI only.** Real PyPI, versioned docs, and the root/stable
  site never move until the **full** release.
- **`skip-existing`** on both publish jobs, so editing / re-firing a release does
  not hard-fail.
- Tags are always cut from **`main`**; each version's docs build from its own
  tagged commit.

### Gotchas

- A GitHub **draft** release triggers nothing — only **Publish** fires CI.
- PyPI *and* TestPyPI are **immutable per version**: a version number can be
  uploaded once and never re-uploaded, even after deletion. If a staged build is
  bad, bump the version.
- Repo setting **Actions → General → Workflow permissions = Read and write** is
  required for the docs `gh-pages` push.
- `release`-triggered workflows always run from the workflow file on the
  **default branch (`main`)**, not from the tag — keep `release.yaml` current on
  `main`.
