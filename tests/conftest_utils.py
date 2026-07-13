import json
import subprocess
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any

EventFile = namedtuple("EventFile", ["path", "count", "metadata"], defaults=[None])

#: The "normal" reference recordings + JSON sidecars: the default tier, shared
#: by the tests *and* the benchmarks so both exercise the exact same data.
RELEASE_URL = "https://github.com/mandulaj/evutils/releases/download/v0.3.14/testfiles.tar.xz"
RELEASE_TAR = "testfiles.tar.xz"
NORMAL_RELEASE_URL = RELEASE_URL  # explicit alias for the tier registry below
NORMAL_RELEASE_TAR = RELEASE_TAR

#: The "small" tier: a ~couple-MB subset (smallest recording per format) for
#: fast branch CI.
SMALL_RELEASE_URL = "https://github.com/mandulaj/evutils/releases/download/v0.3.14/testfiles_small.tar.xz"
SMALL_RELEASE_TAR = "testfiles_small.tar.xz"

#: The "large" tier: an on-demand huge set (never used in CI). TODO: fill URL.
LARGE_RELEASE_URL = "https://github.com/mandulaj/evutils/releases/download/PLACEHOLDER/huge.tar.xz"
LARGE_RELEASE_TAR = "huge.tar.xz"

#: Size tier -> (download URL, tar name, pytest-cache subdir). "normal" keeps
#: the historical "event_files" cache dir so existing caches stay valid.
DATASETS = {
    "small":  (SMALL_RELEASE_URL,  SMALL_RELEASE_TAR,  "event_files_small"),
    "normal": (NORMAL_RELEASE_URL, NORMAL_RELEASE_TAR, "event_files"),
    "large":  (LARGE_RELEASE_URL,  LARGE_RELEASE_TAR,  "event_files_huge"),
}


def download_and_extract_github(url: str, temp_dir: Path, tar_name: str) -> None:
    """Download + extract ``url`` into ``temp_dir`` once.

    A sentinel file records the URL of a completed extraction; on later runs
    the download is skipped if the sentinel matches, so cached data is reused.

    Raises ``RuntimeError`` with a readable message if the download fails or
    returns something that clearly isn't the tarball (e.g. a GitHub "Not
    Found" page), instead of letting ``tar`` fail cryptically downstream.
    """
    marker = temp_dir / ".downloaded"
    if marker.is_file() and marker.read_text().strip() == url:
        return  # already downloaded + extracted this exact release

    tar_path = temp_dir / tar_name
    # -f: fail (non-zero) on HTTP >= 400 instead of saving the error body.
    # -L: follow redirects (release assets 302 to a signed URL).
    # --retry: ride out transient CI network / GitHub hiccups.
    curl = subprocess.run(
        ["curl", "-fL", "--retry", "3", "--retry-delay", "2",
         url, "-o", str(tar_path)],
        capture_output=True, text=True,
    )
    if curl.returncode != 0:
        tar_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download {url} (curl exit {curl.returncode}). "
            f"{curl.stderr.strip()}"
        )

    # A real tarball is many MB; anything tiny is an error page, not data.
    size = tar_path.stat().st_size
    if size < 10_000:
        snippet = tar_path.read_bytes()[:200]
        tar_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download from {url} produced only {size} bytes -- this looks like "
            f"an error response, not the tarball. Check the release URL/asset. "
            f"First bytes: {snippet!r}"
        )

    subprocess.run(["tar", "-xf", str(tar_path), "--strip-components=1", "-C", str(temp_dir)], check=True)
    tar_path.unlink()
    marker.write_text(url)


def fetch_real_event_files(temp_dir: Path, url: str = RELEASE_URL, tar_name: str = RELEASE_TAR) -> dict:
    """Download+extract the reference tarball into ``temp_dir`` (cached), then parse it.

    The single entry point shared by ``tests/conftest.py`` and
    ``benchmarks/conftest.py`` so the two cannot drift.
    """
    download_and_extract_github(url, temp_dir, tar_name)
    return load_event_files(temp_dir)

def register_dataset_option(parser: Any) -> None:
    """Register ``--dataset`` on ``parser``, idempotently.

    The tests and the benchmarks are sibling roots, so their conftests are the
    only place a shared CLI option could live -- but both may be loaded in one
    session (``pytest tests benchmarks``), and pytest would then reject the
    duplicate registration. Both conftests call this; whichever loads first wins
    and the second is a harmless no-op. This keeps the option out of a
    repo-root conftest.
    """
    try:
        parser.addoption(
            "--dataset",
            action="store",
            default="normal",
            choices=["small", "normal", "large"],
            help="Reference-data tier to download (small/normal/large).",
        )
    except ValueError:
        pass  # already registered by the sibling suite's conftest


def fetch_real_event_files_for(size: str, cache: Any) -> dict:
    """Fetch+parse the reference tarball for the given size tier.

    ``size`` is one of ``DATASETS`` ("small"/"normal"/"large"); ``cache`` is the
    pytest ``config.cache`` object. Each tier extracts into its own subdir so the
    tiers never clobber one another. The single entry point both
    ``tests/conftest.py`` and ``benchmarks/conftest.py`` use to honour
    ``--dataset``.
    """
    url, tar_name, subdir = DATASETS[size]
    return fetch_real_event_files(cache.mkdir(subdir), url=url, tar_name=tar_name)


def load_event_files(data_dir: Path) -> dict[str, list[EventFile]]:
    """Parse the JSON descriptions in ``data_dir`` and return
    ``{format: [EventFile, ...]}``.

    Every recording must be accompanied by a ``<name>.json`` sidecar carrying
    at least ``format``, ``filename`` and ``count`` (the reference OpenEB
    event count); reference counts are never hardcoded. Recordings without a
    sidecar are ignored.
    """
    result = defaultdict(list)
    for json_path in sorted(data_dir.glob("*.json")):
        with open(json_path) as f:
            meta = json.load(f)
        path = data_dir / meta["filename"]
        if path.exists():
            result[meta["format"]].append(
                EventFile(path=path, count=meta["count"], metadata=meta)
            )
    return dict(result)
