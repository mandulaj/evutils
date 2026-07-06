import json
import re
import subprocess
import tarfile
import urllib.parse
import urllib.request
from collections import defaultdict, namedtuple
from pathlib import Path, PurePosixPath

EventFile = namedtuple("EventFile", ["path", "count", "metadata"], defaults=[None])

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def download_and_extract_gdrive(file_id: str, temp_dir: Path, tar_name: str) -> None:
    """Download a file from Google Drive, bypassing the virus scan warning if needed, and extract it."""
    tar_file = temp_dir / tar_name
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)

    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        html = response.read().decode('utf-8', errors='ignore')
        form_action_match = re.search(r'action="([^"]+)"', html)
        if form_action_match:
            action_url = form_action_match.group(1)
            inputs = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]+)"', html))
            if not action_url.startswith('http'):
                action_url = 'https://drive.google.com' + action_url
            query = urllib.parse.urlencode(inputs)
            final_url = f"{action_url}?{query}"
            response = urllib.request.urlopen(urllib.request.Request(final_url, headers={'User-Agent': 'Mozilla/5.0'}))

    with open(tar_file, "wb") as f:
        while True:
            chunk = response.read(32768)
            if not chunk:
                break
            f.write(chunk)

    with open(tar_file, "rb") as f:
        magic = f.read(4)
    if magic != _ZSTD_MAGIC:
        tar_file.unlink(missing_ok=True)  # don't leave a poisoned cache entry
        raise RuntimeError(
            f"Downloaded {tar_name} is not a zstd archive (leading bytes {magic!r}) -- "
            "the Google Drive download most likely failed (no network access, or a "
            "quota/consent HTML page). Either allow network access, or download the "
            "recordings on another machine and point EVUTILS_BENCH_DATA at the "
            "directory containing the extracted files + JSON sidecars."
        )

    _extract_tar_zst(tar_file, temp_dir)


def _extract_tar_zst(tar_file: Path, dest: Path) -> None:
    """Extract a ``.tar.zst``, stripping the leading path component.

    Tries, in order: stdlib tarfile (reads zstd natively on Python >= 3.14),
    the ``zstandard`` package, and finally system ``tar`` (which needs zstd
    support -- not the case in some containers, e.g. the OpenEB image).
    """
    try:
        with tarfile.open(tar_file, "r:*") as tf:
            _extract_stripped(tf, dest)
        return
    except tarfile.ReadError:
        pass

    try:
        import zstandard
    except ImportError:
        zstandard = None # type: ignore
    if zstandard is not None:
        with open(tar_file, "rb") as f:
            with zstandard.ZstdDecompressor().stream_reader(f) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tf:
                    _extract_stripped(tf, dest)
        return

    subprocess.run(["tar", "-x", "--strip-components=1", "-C", str(dest), "-f", str(tar_file)], check=True)


def _extract_stripped(tf: "tarfile.TarFile", dest: Path) -> None:
    """Extract regular members of ``tf``, dropping the leading path component."""
    for member in tf:
        if not (member.isfile() or member.isdir()):
            continue
        parts = PurePosixPath(member.name).parts
        if len(parts) <= 1:
            continue
        member.name = str(PurePosixPath(*parts[1:]))
        try:
            tf.extract(member, dest, filter="data")
        except TypeError:  # Python < 3.12: no filter parameter
            tf.extract(member, dest)


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
