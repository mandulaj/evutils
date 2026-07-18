"""Unified event-camera playback / rendering example.

Renders event data as an RGB histogram frame per window and plays it back with
OpenCV. Positive-polarity events are drawn red, negative events blue. A single
render loop drives four interchangeable decoder backends, selected with
``--backend``.

Backend packages are imported lazily inside their own setup branch, so a
backend you do not use (and have not installed) never needs to be importable.

Playback defaults are tuned for *smooth* viewing, not raw throughput: a 33 ms
delta_t window (~30 fps), real-time pacing on, and the synchronous reader. Large
windows (e.g. --window 200000) show few, huge frames and look choppy on any
decoder; async prefetch competes with the numba histogram for the GIL / memory
bandwidth and adds jitter, so it is off by default. For a decode benchmark,
turn pacing off and prefetch on: ``--no-real-time --async-read --no-timing``.

Usage examples
--------------
evutils reader, smooth 30 fps playback (default)::

    python examples/play.py ./data/fan/evt3_fan.raw

evutils reader, 100 ms windows, real-time paced at 2x::

    python examples/play.py ./data/fan/evt3_fan.raw --mode delta_t --window 100000 \\
        --playback-speed 2.0

evutils reader, decode-throughput mode (async prefetch, no pacing)::

    python examples/play.py ./data/fan/evt3_fan.raw --window 200000 \\
        --async-read --no-real-time

evutils reader, fixed count windows::

    python examples/play.py ./data/fan/evt3_fan.raw --mode n_events --window 500000

evutils synchronous streamer (native decoder chunks; windowing not applied)::

    python examples/play.py ./data/fan/evt3_fan.raw --backend evutils-stream

Metavision (OpenEB) EventsIterator::

    python examples/play.py ./data/fan/evt3_fan.raw --backend metavision --window 200000

expelliarmus Wizard (evt3, time-window only)::

    python examples/play.py ./data/fan/evt3_fan.raw --backend expelliarmus --window 33000
"""

import argparse
import time

import cv2
import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Histogram kernels
# ---------------------------------------------------------------------------


@njit
def histogram_soa(x, y, p, width, height):
    """RGB histogram from struct-of-arrays event columns (evutils backends).

    Feeding the SoA columns straight to the njit kernel avoids a per-frame
    ``to_numpy()`` AoS copy of every event -- the single biggest source of
    playback stutter on a dense recording.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(x.shape[0]):
        xi, yi = int(x[i]), int(y[i])
        if 0 <= xi < width and 0 <= yi < height:
            if p[i] == 1:
                img[yi, xi, 0] = 255  # red for positive events
            else:
                img[yi, xi, 2] = 255  # blue for negative events
    return img


@njit
def histogram_aos(events, width, height):
    """RGB histogram from an array-of-structs record array (metavision/expelliarmus)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for event in events:
        x, y = int(event["x"]), int(event["y"])
        if 0 <= x < width and 0 <= y < height:
            if event["p"] == 1:
                img[y, x, 0] = 255  # red for positive events
            else:
                img[y, x, 2] = 255  # blue for negative events
    return img


# ---------------------------------------------------------------------------
# Backend setup
# ---------------------------------------------------------------------------


def build_iterator(args):
    """Build the window iterator for the chosen backend.

    Returns
    -------
    (iterator, layout) : tuple
        ``iterator`` yields one window per step; ``layout`` is ``'soa'`` for the
        evutils backends (access ``ev.x`` / ``ev.t``) or ``'aos'`` for the
        metavision / expelliarmus record-array backends (access ``ev['x']`` /
        ``ev['t']``).
    """
    backend = args.backend

    # Flags that only apply to the evutils reader; warn if explicitly set
    # otherwise. (real_time defaults on, so it is not in this list -- only
    # non-default overrides are worth a note.)
    def _note_ignored_evutils_flags():
        ignored = []
        if args.async_read:
            ignored.append("--async-read")
        if args.playback_speed != 1.0:
            ignored.append("--playback-speed")
        if args.prefetch_depth != 2:
            ignored.append("--prefetch-depth")
        if ignored:
            print(
                f"note: {', '.join(ignored)} apply only to the 'evutils' "
                f"backend; ignored for '{backend}'."
            )

    if backend == "evutils":
        try:
            from evutils.io import EventReader
        except ImportError as e:
            raise ImportError("backend 'evutils' requires the evutils package") from e

        window_kwargs = {}
        if args.mode == "delta_t":
            window_kwargs["delta_t"] = args.window
        else:
            window_kwargs["n_events"] = args.window

        it = EventReader(
            args.file,
            normalize_ts=args.normalize_ts,
            async_read=args.async_read,
            prefetch_depth=args.prefetch_depth,
            reuse_buffers=args.reuse_buffers,
            real_time=args.real_time,
            playback_speed=args.playback_speed,
            max_events=args.max_events,
            **window_kwargs,
        )
        return it, "soa"

    if backend == "evutils-stream":
        try:
            from evutils.io import EventStreamer
        except ImportError as e:
            raise ImportError(
                "backend 'evutils-stream' requires the evutils package"
            ) from e

        _note_ignored_evutils_flags()
        # EventStreamer performs zero slicing: it yields raw decoder chunks in
        # the parser's native block size. It has no delta_t / n_events windowing
        # (kwargs are forwarded to the decoder, which does not window either), so
        # --mode / --window do not apply here.
        print(
            "note: 'evutils-stream' yields native decoder chunks; "
            "--mode / --window are not applied."
        )
        it = EventStreamer(args.file)
        return it, "soa"

    if backend == "metavision":
        try:
            from metavision_core.event_io import EventsIterator
        except ImportError as e:
            raise ImportError(
                "backend 'metavision' requires the metavision_core package "
                "(OpenEB SDK)"
            ) from e

        _note_ignored_evutils_flags()
        if args.mode == "delta_t":
            it = EventsIterator(args.file, delta_t=args.window)
        else:
            it = EventsIterator(args.file, n_events=args.window)
        return it, "aos"

    if backend == "expelliarmus":
        if args.mode != "delta_t":
            raise ValueError(
                "backend 'expelliarmus' supports time windows only "
                "(set_time_window); use --mode delta_t."
            )
        try:
            from expelliarmus import Wizard
        except ImportError as e:
            raise ImportError(
                "backend 'expelliarmus' requires the expelliarmus package"
            ) from e

        _note_ignored_evutils_flags()
        wiz = Wizard(encoding="evt3", chunk_size=1_000_000)
        wiz.set_file(str(args.file))
        wiz.set_time_window(args.window)
        return wiz.read_time_window(), "aos"

    raise ValueError(f"unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# Render loop
# ---------------------------------------------------------------------------


def render(iterator, layout, args):
    """Pull windows from ``iterator`` and play them back as histogram frames."""
    width, height = args.width, args.height
    timing = not args.no_timing

    time0 = time.perf_counter()
    for i, ev in enumerate(iterator):
        time1 = time.perf_counter()
        if len(ev) == 0:
            continue

        if layout == "soa":
            t0 = ev.t[0] / 1_000_000
            frame = histogram_soa(ev.x, ev.y, ev.p, width, height)
        else:
            t0 = ev["t"][0] / 1_000_000
            frame = histogram_aos(ev, width, height)
        time2 = time.perf_counter()

        cv2.putText(
            frame,
            f"t={t0:.3f} s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Histogram", frame)
        time3 = time.perf_counter()
        key = cv2.waitKey(1)
        time4 = time.perf_counter()

        if timing:
            print(
                f"{i} Decode: {1000*(time1-time0):.2f} ms | "
                f"Histogram: {1000*(time2-time1):.2f} ms | "
                f"ImShow: {1000*(time3-time2):.2f} ms | "
                f"WaitKey: {1000*(time4-time3):.2f} ms | "
                f"Total: {1000*(time4-time0):.2f} ms | "
                f"FPS: {1/(time4-time0):.2f}"
            )

        # 'q' or ESC quits; a closed window (X button) also stops playback.
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty("Histogram", cv2.WND_PROP_VISIBLE) < 1:
            break

        time0 = time4

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="event data file",
    )
    parser.add_argument(
        "--backend",
        choices=["evutils", "evutils-stream", "metavision", "expelliarmus"],
        default="evutils",
        help="decoder backend (default: evutils)",
    )

    parser.add_argument(
        "--mode",
        choices=["delta_t", "n_events"],
        default="delta_t",
        help="windowing mode (default: delta_t)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=33_000,
        help="window size: microseconds for delta_t, event "
        "count for n_events (default: 33000 = ~30 fps)",
    )

    # evutils reader tuning (ignored-with-note for other backends)
    parser.add_argument(
        "--async-read",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="async_read",
        help="[evutils] decode ahead on a worker thread; raises throughput "
        "but adds playback jitter (default: off -- smoother for viewing)",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=2,
        help="[evutils] windows to decode ahead when --async-read (default: 2)",
    )
    parser.add_argument(
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="[evutils] pace playback to the recording timeline "
        "(default: on; use --no-real-time to play as fast as decoded)",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="[evutils] real-time speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--reuse-buffers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="[evutils] recycle per-window decode buffers " "(default: on)",
    )
    parser.add_argument(
        "--normalize-ts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="[evutils] normalize timestamps to start at zero " "(default: on)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=50_000_000,
        help="[evutils] max events per window safety cap " "(default: 50000000)",
    )

    parser.add_argument(
        "--width", type=int, default=1280, help="sensor width (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="sensor height (default: 720)"
    )

    parser.add_argument(
        "--no-timing",
        action="store_true",
        default=False,
        help="suppress the per-frame timing printout",
    )

    args = parser.parse_args()

    iterator, layout = build_iterator(args)
    render(iterator, layout, args)


if __name__ == "__main__":
    main()
