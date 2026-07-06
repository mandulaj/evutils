"""Example 1: Basic Reading, Writing, and Format Conversion.

This example demonstrates how to read an event file (like .evt3),
inspect the loaded event data, and write it out to a different
format (like .npz or .csv) effortlessly using `evutils`.
"""

import sys
from pathlib import Path
from evutils.io import EventReader, EventWriter

def main(input_file: str) -> None:
    # 1. Initialize the reader
    # EventReader automatically detects the format based on the file extension
    # or header. We will read the first 1,000,000 events.
    print(f"Reading from: {input_file}")
    reader = EventReader(input_file, n_events=1_000_000)

    # 2. Read the events into memory
    # `events` is an `EventArray` which acts like an Array-of-Structs view
    # over a highly optimized Struct-of-Arrays memory layout.
    events = reader.read()
    print(f"Successfully read {len(events)} events.")

    if len(events) == 0:
        print("No events found in file!")
        return

    # 3. Inspecting the data
    # You can access columns directly: events.t, events.x, events.y, events.p
    t_start = events.t[0]
    t_end = events.t[-1]
    duration_sec = (t_end - t_start) / 1e6
    
    print("\n--- Event Statistics ---")
    print(f"First timestamp: {t_start} µs")
    print(f"Last timestamp:  {t_end} µs")
    print(f"Total duration:  {duration_sec:.3f} seconds")
    print(f"Event rate:      {len(events) / duration_sec:,.0f} events/sec" if duration_sec > 0 else "Event rate: N/A")

    # 4. Slicing data
    # You can slice EventArrays just like numpy arrays. 
    # Let's take only the first 500k events.
    first_half = events[:500_000]
    
    # 5. Format Conversion
    # Let's write these events to a compressed NumPy format (.npz)
    output_file = "converted_events.npz"
    print(f"\nWriting the first {len(first_half)} events to: {output_file}")
    
    with EventWriter(output_file) as writer:
        writer.write(first_half)
        
    print("Conversion complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <input_event_file>")
        print("Example: python 01_read_write_convert.py sample.evt3")
        sys.exit(1)
        
    main(sys.argv[1])
