#!/usr/bin/env python3
"""
Generate metadata JSON files for event files.
Usage: python scripts/generate_metadata.py path/to/file.raw ...
"""
import json
import argparse
from pathlib import Path
from metavision_core.event_io import EventsIterator

import numpy as np

def get_evt_format_version(file_path: str) -> str:
    """Get the version of the evt file format from the file header."""
    # Parse the % headers until 
    with open(file_path, 'rb') as f:
        for line in f:
            if not line.startswith(b'%'):
                break  # End of header
            
            if line.startswith(b'% format '):
                # Example: b'% format EVT3;height=720;width=1280\n'
                # Splits by ';' to isolate the format part, then grabs the last word
                format_val = line.split(b';')[0].split(b' ')[-1]
                return format_val.decode('ascii').lower()
            
            if line.strip() == b'% end':
                break
                
    return ""



def generate_metadata(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"File {path} does not exist.")
        return

    print(f"Processing {path}...")

    # Get the version of the evt file format from the file header


    reader = EventsIterator(str(path))

    height, width = reader.get_size()
    
    total_events = 0
    pos_events = 0
    neg_events = 0
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_t, max_t = float('inf'), float('-inf')

    first_ts = None
    last_ts = None
    for events in reader:
        n = len(events)
        if n == 0:
            continue
        total_events += n
        if first_ts is None:
            first_ts = int(np.min(events['t']))
        
        pos_events += int(np.sum(events['p'] == 1))
        neg_events += int(np.sum(events['p'] == 0))
        
        min_x = int(min(min_x, np.min(events['x'])))
        max_x = int(max(max_x, np.max(events['x'])))
        min_y = int(min(min_y, np.min(events['y'])))
        max_y = int(max(max_y, np.max(events['y'])))
        min_t = int(min(min_t, np.min(events['t'])))
        max_t = int(max(max_t, np.max(events['t'])))
        last_ts = int(np.max(events['t']))
    
    external_triggers = reader.get_ext_trigger_events()
    num_external_triggers = len(external_triggers) if external_triggers is not None else 0
    num_pos_external_triggers = int(np.sum(external_triggers['p'] == 1)) if external_triggers is not None else 0
    num_neg_external_triggers = int(np.sum(external_triggers['p'] == 0)) if external_triggers is not None else 0
    

    metadata = {
        "format": get_evt_format_version(str(path)),
        "filename": path.name,
        "count": total_events,
        "pos_count": pos_events,
        "neg_count": neg_events,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "external_triggers": {
            "total": num_external_triggers,
            "positive": num_pos_external_triggers,
            "negative": num_neg_external_triggers
        },
        "resolution": [width, height],
        "duration_us": int(max_t - min_t) if total_events > 0 else 0
    }

    json_path = path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON metadata for event files.")
    parser.add_argument("files", nargs="+", help="Path to raw/dat event files")
    args = parser.parse_args()
    
    for f in args.files:
        generate_metadata(f)
