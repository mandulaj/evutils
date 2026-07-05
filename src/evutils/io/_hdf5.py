"""HDF5 file decoder and encoder."""

import io
from pathlib import Path
from typing import Union

import h5py
import hdf5plugin
import numba as nb
import numpy as np
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import EventArray

from ..types import Event_dtype
from .common import EventDecoder, EventEncoder


@nb.njit
def get_idx(events, ms_to_idx, last_ms_idx, n_written_events, max_ms, offset):
    """Get the index for each millisecond in the events array.

    Parameters
    ----------
    events : np.ndarray
        Array of events.
    ms_to_idx : np.ndarray
        Array mapping millisecond to index.
    last_ms_idx : int
        Last millisecond index processed.
    n_written_events : int
        Number of events written so far.
    max_ms : int
        Maximum millisecond value in the current chunk.
    offset : int
        Offset for the index calculation.

    Returns
    -------
    None

    """
    idx = 0
    for ms in range(last_ms_idx, max_ms+1):
        while idx < len(events) and events['t'][idx] // 1000 < ms:
            idx += 1

        ms_to_idx[ms] = max(idx + offset + n_written_events, 0)


class EventEncoder_HDF5(EventEncoder):
    def __init__(self, writable: io.BufferedWriter, width:int=1280, height:int=720, chunksize:int=10000):
        """Write events to a HDF5 file.

        Parameters
        ----------
        writable : io.BufferedWriter
            The file-like object to write to
        width : int, optional
            The width of the frame
        height : int, optional
            The height of the frame
        chunksize : int, optional
            The size of the chunks for HDF5 dataset, default 10000

        """
        super().__init__(writable, width=width, height=height)

        self._chunksize = chunksize

        self._ms_to_idx = np.empty(0, dtype=np.uint64)
        self._last_ms_idx = 0



    def init(self):
        """Initialize the HDF5 writer.

        Returns
        -------
        None

        """
        if self._is_initialized:
            return

        self._fd = h5py.File(self._file, "w")
        self._events_group_h5 = self._fd.create_group("events")
        self._compressor = hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

        self._fd.attrs['width'] = self._width
        self._fd.attrs['height'] = self._height

        # Create datasets
        self._events_group_h5.create_dataset("x", shape=(0,), chunks=(self._chunksize, ), maxshape=(None,),
                                            dtype="uint16", **self._compressor)
        self._events_group_h5.create_dataset("y", shape=(0,), chunks=(self._chunksize, ), maxshape=(None,),
                                             dtype="uint16", **self._compressor)
        self._events_group_h5.create_dataset("p", shape=(0,), chunks=(self._chunksize, ), maxshape=(None,),
                                             dtype="uint8", **self._compressor)
        self._events_group_h5.create_dataset("t", shape=(0,), chunks=(self._chunksize, ), maxshape=(None,),
                                             dtype="uint32", **self._compressor)


        self._is_initialized = True

    def write(self, events: np.ndarray) -> int:
        """Write events to the HDF5 file.

        Parameters
        ----------
        events : np.ndarray
            Array of events to write.

        Returns
        -------
        int
            Number of events written.

        """
        if not self._is_initialized:
            self.init()

        # Generate ms_to_idx
        self.__get_ms_idx_for_events(events)

        # Append events
        self.__append_new_events(events)

        return len(events)


    def close(self):
        """Close the HDF5 file and write indexing metadata.

        Returns
        -------
        None

        """
        if not self._is_initialized:
            return

        self._ms_to_idx.append(self._events_group_h5["x"].shape[0])

        # Write the ms_to_idx
        self._fd.create_dataset("ms_to_idx", data=self._ms_to_idx, dtype="uint64", **self._compressor)

        self._fd['ms_to_idx'].resize((len(self._ms_to_idx),))
        self._fd['ms_to_idx'][:] = self._ms_to_idx


        # self._append_new_events(self._buffer)
        self._fd.close()

    def __get_ms_idx_for_events(self, events: np.ndarray, offset=-1):
        max_ms = int(events["t"][-1] // 1000)

        if max_ms + 1 > len(self._ms_to_idx):
            self._ms_to_idx.resize(max_ms + 1, refcheck=False)

        get_idx(events, self._ms_to_idx, self._last_ms_idx, self._n_written_events, max_ms, offset)


        self._last_ms_idx = max_ms + 1





    def __append_new_events(self, events: np.ndarray):
        n_events = events.shape[0]
        x = self._events_group_h5["x"]
        y = self._events_group_h5["y"]
        p = self._events_group_h5["p"]
        t = self._events_group_h5["t"]

        x.resize((x.shape[0] + n_events), axis=0)
        y.resize((y.shape[0] + n_events), axis=0)
        p.resize((p.shape[0] + n_events), axis=0)
        t.resize((t.shape[0] + n_events), axis=0)

        x[-n_events:] = events["x"]
        y[-n_events:] = events["y"]
        p[-n_events:] = events["p"]
        t[-n_events:] = events["t"]

        self._n_written_events += n_events



class EventDecoder_HDF5(EventDecoder):
    """Read events from a HDF5 file.

    Parameters
    ----------
    file : str
        The file to read from
    width : int, optional
        The width of the frame
    height : int, optional
        The height of the frame

    """

    def __init__(self, file:Union[Path, str]):
        super().__init__(file)


    def init(self):
        """Initialize the HDF5 reader.

        Returns
        -------
        None

        """
        if self._is_initialized:
            return
        self._fd = h5py.File(self._file, "r")
        self._ms_to_idx = np.asarray(self._fd["ms_to_idx"])
        self._max_events = self._fd["events"]["x"].shape[0]
        self._last_ms = len(self._ms_to_idx) - 1
        self._is_initialized = True

    def read(self, start_ms: int = 0, end_ms: int = -1) -> 'EventArray':
        """Read events from a specific time range.

        Parameters
        ----------
        start_ms : int, optional
            Start time in milliseconds, by default 0.
        end_ms : int, optional
            End time in milliseconds, by default -1 (until the end).

        Returns
        -------
        events : EventArray
            Array of events within the time range.

        """
        if not self._is_initialized:
            self.init()
        if start_ms > len(self._ms_to_idx) - 1:
            print(f"Start time {start_ms} is greater than the highest available ms time {len(self._ms_to_idx) - 1}")
            from ..types import EventArray
            return EventArray.empty()
        if end_ms == -1 or end_ms > self._last_ms:
            end_ms = self._last_ms

        assert start_ms >= 0, "start_ms must be greater or equal to 0"
        assert start_ms <= end_ms, "start_ms must be smaller than end_ms"

        start_idx = self._ms_to_idx[start_ms]
        end_idx = self._ms_to_idx[end_ms]
        return self._read_events(start_idx, end_idx)

    def _read_events(self, start_idx: int, end_idx: int) -> 'EventArray':
        """Read events from a specific index range.

        Parameters
        ----------
        start_idx : int
            Start index.
        end_idx : int
            End index.

        Returns
        -------
        events : EventArray
            Array of events within the index range.

        """
        x = self._fd["events"]["x"][start_idx:end_idx]
        y = self._fd["events"]["y"][start_idx:end_idx]
        p = self._fd["events"]["p"][start_idx:end_idx]
        t = self._fd["events"]["t"][start_idx:end_idx]
        from ..types import EventArray
        return EventArray(t, x, y, p)
