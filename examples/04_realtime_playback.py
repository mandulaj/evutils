"""Example 4: Real-Time Playback with a Live Histogram View.

Example 3 accumulates frames as fast as the file can be decoded. Here we
replay the recording at its *original* speed instead: with
``real_time=True`` the EventReader paces every chunk against a wall-clock
anchor, so the stream arrives exactly as it did from the live sensor.

Key points demonstrated:
- ``real_time=True``: chunks are released on the recording's own timeline.
- ``playback_speed``: 2.0 doubles the speed, 0.5 gives slow motion.
- ``async_read=True``: the next window is decoded in a background thread
  while we sleep out the pacing delay and render the current frame - the
  decode time hides entirely inside the playback delay.
- Pacing is anchored absolutely: however long our histogram + drawing takes,
  the reader compensates automatically. If processing is slower than the
  recording, no artificial delay is added at all.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from evutils.io import EventReader
from evutils.dense import histogram


def main(input_file: str, playback_speed: float = 1.0) -> None:
    # Time window: 33 milliseconds per frame (~30 FPS at 1x speed)
    frame_duration_us = 33_000

    fallback_width, fallback_height = 1280, 720

    print(f"Reading: {input_file}")
    print(f"Playback speed: {playback_speed}x "
          f"({frame_duration_us / 1000 / playback_speed:.1f} ms of wall time per frame)\n")

    with EventReader(
        input_file,
        delta_t=frame_duration_us,
        real_time=False,                 # pace chunks like a live sensor
        playback_speed=playback_speed,  # 1.0 = original recording speed
        async_read=False,                # decode ahead while we wait/draw
    ) as reader:

        try:
            width, height = reader.shape()
            print(f"Detected resolution from metadata: {width}x{height}")
        except (NotImplementedError, ValueError):
            width, height = fallback_width, fallback_height
            print(f"No resolution found in metadata. Assuming {width}x{height}")

        # Warm up the one-time costs BEFORE playback starts, or they show up
        # as lag on the first frame: `histogram` is numba-compiled on its
        # first call (can take a second), and the first `imshow` creates the
        # window. The reader's pacing anchor is only set when the first chunk
        # is delivered, so doing this now keeps the playback clock honest.
        from evutils.types import Event_dtype
        histogram(np.zeros(1, dtype=Event_dtype), width=width, height=height,
                  dtype=np.uint8, fill=True)
        window = "Real-time playback (q to quit)"
        cv2.imshow(window, np.zeros((height, width), dtype=np.uint8))
        cv2.waitKey(0)

        wall_start = time.perf_counter()
        first_ts: int | None = None

        for frame_idx, events in enumerate(reader):
            if len(events) == 0:
                continue
            if first_ts is None:
                first_ts = int(events.t[0])

            limit_ev = min(len(events), 100_000)

            frame = histogram(events.to_numpy()[:limit_ev], width=width, height=height,
                              dtype=np.uint8, fill=True)

            # How closely does the playback clock track the recording clock?
            # (lag ~0 means the pacing keeps up; positive lag means decode or
            # rendering is slower than real time and playback runs behind.)
            recording_s = (int(events.t[-1]) - first_ts) / 1e6 / playback_speed
            wall_s = time.perf_counter() - wall_start
            lag_ms = (wall_s - recording_s) * 1e3

            print(f"Frame {frame_idx:03d} | "
                  f"Events: {len(events):7,d} | "
                  f"Recording clock: {recording_s:6.2f} s | "
                  f"Wall clock: {wall_s:6.2f} s | "
                  f"Lag: {lag_ms:+6.1f} ms")

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

    cv2.destroyAllWindows()
    print("\nPlayback finished!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <input_event_file> [playback_speed]")
        print("Example: python 04_realtime_playback.py sample.evt3      # original speed")
        print("Example: python 04_realtime_playback.py sample.evt3 2.0  # twice as fast")
        sys.exit(1)
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    main(sys.argv[1], speed)
