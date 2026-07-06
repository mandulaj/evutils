"""Example 3: Accumulating Events into Frames.

Standard Computer Vision algorithms and Convolutional Neural Networks (CNNs)
expect 2D frames (images) instead of sparse asynchronous event streams.
This example demonstrates how to stream an event file using fixed time-windows
(`delta_t`) and accumulate those events into 2D histogram frames.
"""

import sys
from pathlib import Path
import numpy as np
from evutils.io import EventReader
import cv2

from evutils.repr import histogram

def main(input_file: str) -> None:
    # Time window: 33 milliseconds per frame (~30 FPS)
    frame_duration_us = 33_000 
    
    # We don't always know the sensor resolution upfront, but the EventReader
    # usually exposes it if the format contains a header (e.g. EVT3, DAT).
    # Otherwise, we can provide a fallback like 1280x720.
    fallback_width, fallback_height = 1280, 720
    
    print(f"Reading: {input_file}")
    print(f"Time window per frame: {frame_duration_us / 1000:.1f} ms\n")

    # Use `delta_t` instead of `n_events` to read time-based chunks!
    with EventReader(input_file, delta_t=frame_duration_us) as reader:
        
        # Try to get the resolution from the file metadata
        try:
            width, height = reader.shape()
            print(f"Detected resolution from metadata: {width}x{height}")
        except (NotImplementedError, ValueError):
            width, height = fallback_width, fallback_height
            print(f"No resolution found in metadata. Assuming {width}x{height}")

        frame_count = 0
        
        for chunk_idx, events in enumerate(reader):
            if len(events) == 0:
                continue
                
            # Accumulate events into a 2D histogram frame
            frame = histogram(events.to_numpy(), width=width, height=height, dtype=np.uint8, fill=True)
            
           
            # (Optional) If you wanted to separate ON and OFF polarities into 2 channels:
            # frame_on = np.zeros((height, width), dtype=np.int32)
            # frame_off = np.zeros((height, width), dtype=np.int32)
            # mask_on = (events.p == 1)
            # np.add.at(frame_on, (events.y[mask_on], events.x[mask_on]), 1)
            # np.add.at(frame_off, (events.y[~mask_on], events.x[~mask_on]), 1)
            
            max_events_in_pixel = frame.max()
            non_zero_pixels = np.count_nonzero(frame)
            
            print(f"Frame {chunk_idx:03d} | "
                  f"Events: {len(events):7,d} | "
                  f"Active Pixels: {non_zero_pixels:7,d} | "
                  f"Max events in single pixel: {max_events_in_pixel}")
            
            frame_count += 1
            
            cv2.imshow("Accumulated Frame", frame.astype(np.uint8)) 
            cv2.waitKey(1)  # Display the frame for 1 ms
            # Let's just process the first 10 frames for this example
            if frame_count >= 10:
                print("Stopping early after 10 frames.")
                break
    cv2.destroyAllWindows()
    print("\nFrame accumulation complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <input_event_file>")
        print("Example: python 03_frame_accumulation.py sample.evt3")
        sys.exit(1)
        
    main(sys.argv[1])
