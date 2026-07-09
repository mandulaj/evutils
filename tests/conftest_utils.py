import json
import subprocess
from collections import defaultdict, namedtuple
from pathlib import Path

EventFile = namedtuple("EventFile", ["path", "count", "metadata"], defaults=[None])

#: Default reference recordings + JSON sidecars. Shared by the tests *and* the
#: benchmarks (default dataset) so both exercise the exact same data.
RELEASE_URL = "https://github.com/mandulaj/evutils/releases/download/v0.3.14/testfiles.tar.xz"
RELEASE_TAR = "testfiles.tar.xz"

#: Optional larger benchmark dataset (GitHub release). TODO: fill in the real URL.
LARGE_RELEASE_URL = "https://github.com/mandulaj/evutils/releases/download/PLACEHOLDER/huge.tar.xz"
LARGE_RELEASE_TAR = "huge.tar.xz"


def download_and_extract_github(url: str, temp_dir: Path, tar_name: str) -> None:
    """Download + extract ``url`` into ``temp_dir`` once.

    A sentinel file records the URL of a completed extraction; on later runs
    the download is skipped if the sentinel matches, so cached data is reused.
    """
    marker = temp_dir / ".downloaded"
    if marker.is_file() and marker.read_text().strip() == url:
        return  # already downloaded + extracted this exact release

    tar_path = temp_dir / tar_name
    subprocess.run(["curl", "-L", url, "-o", str(tar_path)], check=True)
    subprocess.run(["tar", "-x", "--strip-components=1", "-C", str(temp_dir), "-f", str(tar_path)], check=True)
    tar_path.unlink()
    marker.write_text(url)


def fetch_real_event_files(temp_dir: Path, url: str = RELEASE_URL, tar_name: str = RELEASE_TAR) -> dict:
    """Download+extract the reference tarball into ``temp_dir`` (cached), then parse it.

    The single entry point shared by ``tests/conftest.py`` and
    ``benchmarks/conftest.py`` so the two cannot drift.
    """
    download_and_extract_github(url, temp_dir, tar_name)
    return load_event_files(temp_dir)

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
