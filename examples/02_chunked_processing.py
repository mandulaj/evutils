"""Example 2: Streaming and Chunked Processing.

Event files can be massively large (10+ GB). Reading them completely
into memory is often impossible. `evutils` handles this effortlessly
by allowing you to stream through the file in chunks.

This example computes the average event rate and the global
center of mass of the event stream without running out of memory.
"""

import sys
from pathlib import Path
import numpy as np
from evutils.io import EventReader

def main(input_file: str) -> None:
    # 1. Initialize the reader for chunked access
    # We will read chunks of 500,000 events at a time.
    # Alternatively, you can use `delta_t=100_000` to read chunks
    # representing 100ms time windows!
    chunk_size = 500_000
    
    print(f"Streaming: {input_file}")
    print(f"Chunk size: {chunk_size} events\n")

    total_events = 0
    t_start = -1
    t_end = -1
    
    sum_x = 0.0
    sum_y = 0.0

    # 2. Iterate through the file
    # The reader acts as a python generator, yielding EventArrays
    with EventReader(input_file, n_events=chunk_size) as reader:
        for chunk_idx, events in enumerate(reader):
            if len(events) == 0:
                break
                
            if chunk_idx == 0:
                t_start = events.t[0]
            t_end = events.t[-1]
            
            n = len(events)
            total_events += n
            
            # Accumulate spatial sums for Center of Mass
            # We cast to float to prevent integer overflow
            sum_x += np.sum(events.x.astype(np.float64))
            sum_y += np.sum(events.y.astype(np.float64))

            print(f"Processed chunk {chunk_idx:03d} | Total Events so far: {total_events:10,d}")

    # 3. Final calculations
    if total_events == 0:
        print("No events processed.")
        return

    duration_sec = (t_end - t_start) / 1e6
    avg_rate = total_events / duration_sec if duration_sec > 0 else 0
    
    center_of_mass_x = sum_x / total_events
    center_of_mass_y = sum_y / total_events

    print("\n--- Final Stream Statistics ---")
    print(f"Total Events:   {total_events:,}")
    print(f"Total Duration: {duration_sec:.2f} seconds")
    print(f"Avg Event Rate: {avg_rate:,.0f} events/sec")
    print(f"Center of Mass: X={center_of_mass_x:.1f}, Y={center_of_mass_y:.1f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <input_event_file>")
        print("Example: python 02_chunked_processing.py sample.evt3")
        sys.exit(1)
        
    main(sys.argv[1])
