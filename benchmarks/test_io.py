import gc
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from evutils.io import EventReader, EventWriter

import sys
sys.path.append(str(Path(__file__).parent.parent / "tests"))

def _count_pos_events(chunk) -> int:
    return int(np.count_nonzero(chunk.p == 1))

def get_expelliarmus_wizard(fmt: str):
    try:
        from expelliarmus import Wizard
        return Wizard(encoding=fmt)
    except ImportError:
        pytest.skip("expelliarmus not available")

def get_evlib():
    try:
        import evlib
        return evlib
    except ImportError:
        pytest.skip("evlib not available")

@pytest.mark.parametrize("fmt", ["evt3", "evt21", "evt2", "dat", "aer", "npz", "hdf5", "csv", "aedat4"])
@pytest.mark.parametrize("library", ["evutils", "expelliarmus", "evlib"])
@pytest.mark.parametrize("slicing", ["chunk_size", "delta_t"])
def test_read(benchmark, reference_events, fmt, library, slicing):
    """
    Benchmark reading events.
    """
    # Exclude invalid combinations
    if library == "expelliarmus" and fmt not in ["evt2", "evt3", "dat"]:
        pytest.skip("expelliarmus does not support this format")
    if library == "evlib" and fmt not in ["hdf5", "aedat4", "evt3"]:
        pytest.skip("evlib supports only hdf5/aedat4 for reliable decoding in this suite")
        
    benchmark.group = f"read-{fmt}-{slicing}"
    
    # 1. Prepare tmpfs path
    shm_dir = Path("/dev/shm")
    if not shm_dir.exists():
        shm_dir = Path(tempfile.gettempdir())
    
    ext = "raw" if fmt.startswith("evt") else fmt
    test_file = shm_dir / f"bench_read.{ext}"
    
    # 2. Dynamic Transcoding
    # AEDAT4 requires special synth logic, but evutils supports writing most formats
    if fmt == "aedat4":
        from aedat_synth import make_aedat4
        test_file.write_bytes(
            make_aedat4(reference_events["t"], reference_events["x"], reference_events["y"], reference_events["p"], events_per_packet=65536)
        )
    elif fmt == "aer":
        aer = reference_events.copy()
        aer["x"] &= 0x1FF
        aer["y"] &= 0x1FF
        with EventWriter(test_file) as w:
            w.write(aer)
    else:
        writer_kwargs = {"width": 1280, "height": 720}
        if fmt.startswith("evt"):
            writer_kwargs["format"] = fmt
        with EventWriter(test_file, **writer_kwargs) as w:
            w.write(reference_events)

    # 3. Setup caching & GC
    def setup():
        gc.collect()
        with open(test_file, "rb") as f:
            while f.read(16 * 1024 * 1024):
                pass

    # 4. Benchmarking Logic
    if library == "evutils":
        kwargs = {"async_read": True}
        if slicing == "chunk_size":
            kwargs["chunk_size"] = 1_000_000
        else:
            kwargs["delta_t"] = 50_000
            
        def bench_fn():
            total = 0
            pos = 0
            with EventReader(test_file, **kwargs) as reader:
                for chunk in reader:
                    total += len(chunk)
                    pos += _count_pos_events(chunk)
            return total
            
    elif library == "expelliarmus":
        wiz = get_expelliarmus_wizard(fmt)
        def bench_fn():
            wiz.set_file(str(test_file))
            total = 0
            pos = 0
            if slicing == "chunk_size":
                # Expelliarmus doesn't explicitly parameterize chunk_size via kwargs in the iterator easily,
                # but read_chunk() uses the default block size.
                for chunk in wiz.read_chunk():
                    total += len(chunk)
                    pos += int(np.count_nonzero(chunk['p'] == 1))
            else:
                for chunk in wiz.read_time_window(50_000):
                    total += len(chunk)
                    pos += int(np.count_nonzero(chunk['p'] == 1))
            return total

    elif library == "evlib":
        if slicing != "chunk_size":
            pytest.skip("evlib time slicing is not cleanly comparable via streaming here")
        evlib_module = get_evlib()
        import polars as pl
        def bench_fn():
            try:
                df = evlib_module.load_events(str(test_file))
                if hasattr(df, "collect"):
                    return int(df.select(pl.len()).collect(engine="streaming").item())
                return len(df)
            except Exception as e:
                pytest.skip(f"evlib failed: {e}")

    try:
        n = benchmark.pedantic(bench_fn, setup=setup, rounds=4, iterations=1, warmup_rounds=1)
        assert n > 0
        benchmark.extra_info.update(library=library, n_events=n, fmt=fmt, slicing=slicing)
    finally:
        # Cleanup RAM disk
        if test_file.exists():
            test_file.unlink()

@pytest.mark.parametrize("fmt", ["evt3", "evt21", "evt2", "dat", "aer", "npz", "hdf5", "csv", "aedat4"])
@pytest.mark.parametrize("library", ["evutils", "expelliarmus", "evlib"])
def test_write(benchmark, reference_events, fmt, library):
    if library == "expelliarmus" and fmt != "dat":
        pytest.skip("expelliarmus write bench only supports dat")
    if library == "evlib":
        pytest.skip("evlib write bench not cleanly comparable here")
        
    benchmark.group = f"write-{fmt}"
    
    shm_dir = Path("/dev/shm")
    if not shm_dir.exists():
        shm_dir = Path(tempfile.gettempdir())
    ext = "raw" if fmt.startswith("evt") else fmt
    out_file = shm_dir / f"bench_write.{ext}"
    
    def setup():
        gc.collect()

    if library == "evutils":
        ev = reference_events
        
        # AEDAT4 requires special synth logic
        if fmt == "aedat4":
            from aedat_synth import make_aedat4
            def bench_fn():
                out_file.write_bytes(make_aedat4(ev["t"], ev["x"], ev["y"], ev["p"], events_per_packet=65536))
        else:
            if fmt == "aer":
                ev = ev.copy()
                ev["x"] &= 0x1FF
                ev["y"] &= 0x1FF
            
            writer_kwargs = {"width": 1280, "height": 720}
            if fmt.startswith("evt"):
                writer_kwargs["format"] = fmt
                
            def bench_fn():
                with EventWriter(out_file, **writer_kwargs) as w:
                    w.write(ev)
                
    elif library == "expelliarmus":
        wiz = get_expelliarmus_wizard(fmt)
        def bench_fn():
            wiz.save(str(out_file), reference_events)

    try:
        benchmark.pedantic(bench_fn, setup=setup, rounds=4, iterations=1, warmup_rounds=1)
        benchmark.extra_info.update(library=library, n_events=len(reference_events), fmt=fmt, slicing="write")
    finally:
        if out_file.exists():
            out_file.unlink()
