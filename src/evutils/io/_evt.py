

import io
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import numba as nb
import numpy as np

from ..types import Event_dtype, Trigger_dtype
from ._common import EventDecoder, EventEncoder




class EventDecoder_EVT(EventDecoder):
    '''
    Class for reading EVT files from Prophesee cameras

    Parameters
    ----------
    file : str
        Path to the EVT file
    delta_t : int, optional
        Time window in microseconds, by default None
    n_events : int, optional
        Number of events to read, by default None
    max_events : int, optional
        Maximum number of events to read at once, by default 10000000
    mode : {"auto", "delta_t", "n_events", "mixed", "all" }, optional
        Mode of operation, by default "auto"
    buffer_size : int, optional
        Size of the buffer to read events, by default 1_000_000

    Returns
    -------
    out : EventReader_RAW
        EventReader_RAW instance

    Raises
    ------
    ValueError
        If the format in the header is not supported

    Notes
    -----
    The class supports EVT3, EVT2.1 and EVT2 formats

    References
    ----------
    [1] Prophesee RAW file format documentation https://docs.prophesee.ai/stable/data/file_formats/raw.html#chapter-data-file-formats-raw

    '''
    MAX_EVENTS_READ = 1e12
    MAX_DELTA_T = 1e12
    FORMATS = {"evt3": "evt 3.0", "evt21": "evt 2.1", "evt2": "evt 2"}
    EVT_FORMATS = {"3.0": "evt3", "2.1": "evt21", "2.0": "evt2"}

    def __init__(self, readable:io.BufferedReader, chunk_size:int=10_000_000):
        super().__init__(readable=readable, chunk_size=chunk_size)

        # EVT specific variables
        self.last_ts_high_high = 0
        self.last_ts_high = 0
        self.last_ts_low = 0
        self.ts_initialized = False
        self.last_y = 0
        self.format = None

        self.events_buffer = np.empty(self.chunk_size, dtype=Event_dtype)
        self.events_buffer_len = 0
        self.triggers_buffer = np.empty(self.chunk_size, dtype=Trigger_dtype)
        self.triggers_buffer_len = 0

    def init(self):

        # Try to find the end of the header
        is_header = True

        self.header = {
            "date": datetime.now(),
            "evt": None,
            "format": None,
            "generation": None,
            "serial_number": "00000000",
            "system_id": 49,
            "camera_integrator_name": "Prophesee",
            "integrator_name": "Prophesee",
            "sensor_name": None,
            "sensor_generation": None,
            "geometry": None,
            "plugin_name": None,
            "plugin_integrator_name": None,
        }



        while is_header:
            pos = self.fd.tell()
            line = self.fd.readline()
            if line.startswith(b"% end"):
                is_header = False
                break
            if not line.startswith(b"%"):
                is_header = False
                self.fd.seek(pos)
                break

            split = line.decode('utf-8').strip().split(" ")
            key = split[1].lower()
            value = " ".join(split[2:])

            try:
                if key in self.header.keys():
                    if key == "date":
                        value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    elif key == "height" or key == "width" or key == "system_id":
                        value = int(value)

                    self.header[key] = value
                else:
                    print(f"Unknown key {key} in header")
                    print(f"Supported keys are: {list(self.header.keys())}")
                    print(f"Line: {line}")
            except ValueError:
                print(f"Error parsing line: {line}")


        # Check Format, height and width:
        if self.header["format"] is not None:
            split: List[str] = self.header["format"].split(";")
            for s in split:
                if s.startswith("height"):
                    self.height = int(s.split("=")[1])
                elif s.startswith("width"):
                    self.width = int(s.split("=")[1])
                else:
                    s = s.lower().replace(".", "")
                    if s in EventFileReader_RAW.FORMATS.keys():
                        self.format = s
                    else:
                        print(f"Unknown format {s}, supported formats are {list(EventFileReader_RAW.FORMATS.keys())}")

        if self.header["geometry"] is not None:
            split = self.header["geometry"].split("x")
            if len(split) != 2:
                raise ValueError(f"Invalid geometry {self.header['geometry']}")

            if self.width is not None and self.height is not None:
                if self.width != int(split[0]) or self.height != int(split[1]):
                    print(f"Warning: Geometry in header {self.header['geometry']} does not match width/height")
                    print(f"Setting width/height to {split[0]}x{split[1]}")
            self.width = int(split[0])
            self.height = int(split[1])

        if self.header['evt'] is not None:

            if self.header['evt'] in EventFileReader_RAW.EVT_FORMATS.keys():
                format = EventFileReader_RAW.EVT_FORMATS[self.header['evt']]

                if self.format is not None and self.format != format:
                    print(f"Warning: evt in header {self.header['evt']} does not match evt")
                    print(f"Setting evt to {format}")
                    self.format = format


            else:
                print(f"Unknown evt version {self.header['evt']}")
                print(f"Supported evt versions are: {list(EventFileReader_RAW.EVT_FORMATS.keys())}")


        if self.format is None:
            print(f"Error: Format not found in header, trying to use EVT3")
            self.format = "evt3"

        if self.width is None or self.height is None:
            if self.header['sensor_name'] == "IMX636":
                self.width = 1280
                self.height = 720

        if self.width is None or self.width < 0 or self.width > 2048:
            print(f"Error: Valid Width not found in header, setting to 2048")
            self.width = 2048
        if self.height is None or self.height < 0 or self.height > 2048:
            print(f"Error: Valdi Height not found in header, setting to 2048")
            self.height = 2048

        self.is_initialized = True


    def read_chunk(self, delta_t_hint:int = None, n_events_hint:int = None) -> np.ndarray:
        if not self.is_initialized:
            self.init()

        events, triggers = self.__read_and_parse_buffer()
        return events


    def __read_and_parse_buffer(self):
        # print(f"Reading buffer of size {self.buffer_size}")
        assert self.fd is not None

        if self.format == "evt3":

            # Read the buffer as uint16
            input_buffer = np.fromfile(self.fd, dtype=np.uint16, count=self.chunk_size)

            # We have reached the end of the file, but not nessarily the end of the buffer
            if len(input_buffer) < self.chunk_size:
                self.eof = True

            if len(input_buffer) == 0:
                return np.array([], dtype=Event_dtype), np.array([], dtype=Trigger_dtype)

            n_events, n_triggers, self.last_ts_high_high, self.last_ts_high, self.last_ts_low, self.last_y, self.ts_initialized, msg_processed = parse_evt3_buffer(
                    input_buffer,
                    self.events_buffer,
                    self.triggers_buffer,
                    self.last_ts_high_high,
                    self.last_ts_high,
                    self.last_ts_low,
                    self.last_y,
                    self.ts_initialized)

            # self.events_buffer_len += n_events
            # self.triggers_buffer_len += n_triggers
            if msg_processed < len(input_buffer):
                # This happens if We are in the middle of a Vect message and we need to fetch more data
                missed_bytes = 2 * (len(input_buffer) - msg_processed)
                # print(f"Missed {missed_bytes} bytes, seeking back")
                self.fd.seek(-missed_bytes, 1)



            del input_buffer
        elif self.format == "evt21":
            raise NotImplementedError("EVT2.1 not implemented")
        elif self.format == "evt2":
            raise NotImplementedError("EVT2 not implemented")
        else:
            raise ValueError(f"Unsupported format {self.format}. Supported formats are {list(EventFileReader_RAW.FORMATS.keys())}")
        return self.events_buffer[:n_events], self.triggers_buffer[:n_triggers]


    def reset(self):
        assert self.fd is not None
        self.fd.seek(0)
        self.is_initialized = False
        self.eof = False




class EventEncoder_EVT(EventEncoder):
    '''
    Class for writing EVT files from Prophesee cameras

    Parameters
    ----------
    file : str
        Path to the RAW file
    width : int, optional
        Width of the frame, by default 1280
    height : int, optional
        Height of the frame, by default 720
    dt : datetime, optional
        Timestamp of the recording (default is the current time)
    serial : str, optional
        Serial number of the camera, by default "00000000"
    format : {"evt3", "evt21", "evt2"}
        Format of the file, by default "evt3"

    Raises
    ------
    ValueError
        If the format is not supported

    Notes
    -----
    The class supports EVT3, EVT2.1 and EVT2 formats

    References
    ----------
    [1] Prophesee RAW file format documentation https://docs.prophesee.ai/stable/data/file_formats/raw.html#chapter-data-file-formats-raw

    Examples
    --------

    >>> writer = EventWriter_RAW("events.raw", width=1280, height=720, dt=datetime.now(), serial="00000000", format="evt3")
    >>> writer.write(events)

    '''
    FORMATS = {"evt3": "evt 3.0", "evt21": "evt 2.1", "evt2": "evt 2"}
    def __init__(self, writable: io.BufferedWriter, width:int=1280, height:int=720, dt:datetime|None=None, serial:str="00000000", format:str="evt3"):
        super().__init__(writable, width, height, dt)

        format = format.lower().replace(".", "")
        if format not in EventFileWriter_RAW.FORMATS.keys():
            raise ValueError(f"Unsupported format {format}. Supported formats are {list(EventFileWriter_RAW.FORMATS.keys())}")
        self.format = format

        self.system_id = 49

        self.last_upper12_ts = -1
        self.last_lower12_ts = -1
        self.last_y  = -1

        self.serial_number = serial

        self.formatted_datetime = self.dt.strftime("%Y-%m-%d %H:%M:%S")

    def init(self):
        if self.is_initialized:
            return

        self.fd.write(
f"""% camera_integrator_name Prophesee
% date {self.formatted_datetime}
% {self.FORMATS[self.format]}
% format {self.format};height={self.height};width={self.width}
% generation 4.2
% geometry {self.width}x{self.height}
% integrator_name Prophesee
% plugin_integrator_name Prophesee
% plugin_name hal_plugin_prophesee
% sensor_generation 4.2
% serial_number {self.serial_number}
% system_ID {self.system_id}
% end
""".encode('utf-8'))
        self.is_initialized = True


    def write(self, events: np.ndarray) -> int:
        assert self.fd is not None

        if not self.is_initialized:
            self.init()

        if self.format == "evt3":
            buffer, self.last_lower12_ts, self.last_upper12_ts, self.last_y = get_raw_evt3_buffer(
                events,
                self.last_lower12_ts,
                self.last_upper12_ts,
                self.last_y)
        elif self.format == "evt21":
            raise NotImplementedError("EVT2.1 not implemented")
        elif self.format == "evt2":
            raise NotImplementedError("EVT2 not implemented")
        else:
            raise NotImplementedError(f"Unsupported format {self.format}. Supported formats are {list(EventFileWriter_RAW.FORMATS.keys())}")

        self.n_written_events += len(events)

        buffer.tofile(self.fd)

        return len(events)
