import json
import re
import subprocess
import tarfile
import urllib.parse
import urllib.request
from collections import defaultdict, namedtuple
from pathlib import Path, PurePosixPath

EventFile = namedtuple("EventFile", ["path", "count", "metadata"], defaults=[None])


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
