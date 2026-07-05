import json
import re
import subprocess
import urllib.parse
import urllib.request
from collections import defaultdict, namedtuple
from pathlib import Path

EventFile = namedtuple("EventFile", ["path", "count", "metadata"], defaults=[None])

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
            
    subprocess.run(["tar", "-x", "--strip-components=1", "-C", str(temp_dir), "-f", str(tar_file)], check=True) 


def load_event_files(temp_dir: Path, expected_paths: dict[str, Path] = None) -> dict[str, list[EventFile]]:
    """Parse JSON metadata in temp_dir and return a dict of {format: [EventFile, ...]}.
    
    If no JSONs are found, fallback to hardcoded counts for expected_paths (used for 'small' dataset).
    """
    result = defaultdict(list)
    json_files = list(temp_dir.glob("*.json"))
    
    if json_files:
        for json_path in json_files:
            with open(json_path) as f:
                meta = json.load(f)
                fmt = meta["format"]
                path = temp_dir / meta["filename"]
                if path.exists():
                    result[fmt].append(EventFile(path=path, count=meta["count"], metadata=meta))
    else:
        # Fallback to hardcoded counts if no JSON files found (small dataset)
        if expected_paths:
            event_counts = {
                'evt3': 33494595,
                'evt21': 8214341,
                'evt2': 33494595,
            }
            for fmt, path in expected_paths.items():
                if path.exists():
                    result[fmt].append(EventFile(path=path, count=event_counts[fmt], metadata=None))
                    
    return dict(result)
