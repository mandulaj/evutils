# Releasing evutils

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
