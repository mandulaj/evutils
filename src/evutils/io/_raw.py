

import io
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import numba as nb
import numpy as np

from ..types import Event_dtype, Trigger_dtype
from .common import EventDecoder, EventEncoder

EVT3_EVT_ADDR_Y = 0x0000
EVT3_EVT_ADDR_X = 0x2000
EVT3_VECT_BASE_X = 0x3000
EVT3_VECT_12 = 0x4000
EVT3_VECT_8 = 0x5000
EVT3_EVT_TIME_LOW = 0x6000
EVT3_CONTINUED_4 = 0x7000
EVT3_EVT_TIME_HIGH = 0x8000
EVT3_EXT_TRIGGER = 0xA000
EVT3_OTHERS = 0xE000
EVT3_CONTINUED_12 = 0xF000



@nb.njit
def get_raw_evt3_buffer(events: np.ndarray, last_lower12_ts: int, last_upper12_ts: int, last_y: int, master = True):
    # Pre-allocate large buffer
    buffer = np.zeros(len(events) * 8, dtype=np.uint8)

    # Prepare the master/slave bit
    if master:
        master_slave = 0x000
    else:
        master_slave = 0x800

    # Current position of the buffer
    i = 0

    for ev in events:
        upper12_ts = (int(ev['t']) & 0x0FFF000) >> 12
        lower12_ts = int(ev['t']) & 0x00000FFF

        # EVT_TIME_HIGH - Updates the higher 12-bit portion of the 24-bit time base ‘1000’
        if upper12_ts != last_upper12_ts:
            last_upper12_ts = upper12_ts
            value = EVT3_EVT_TIME_HIGH | (upper12_ts & 0xFFF)

            buffer[i] = value & 0xFF
            buffer[i+1] = (value >> 8) & 0xFF
            i += 2

        # EVT_TIME_LOW - Updates the lower 12-bit portion of the 24-bit time base ‘0110’
        if lower12_ts != last_lower12_ts:
            last_lower12_ts = lower12_ts
            value = EVT3_EVT_TIME_LOW | (lower12_ts & 0xFFF)

            buffer[i] = value & 0xFF
            buffer[i+1] = (value >> 8) & 0xFF
            i += 2

        # EVT_ADDR_Y - Y coordinate, and system type (master/slave camera) ‘0000’
        if last_y != ev['y']:
            last_y = ev['y']
            value = (EVT3_EVT_ADDR_Y | master_slave | (int(ev['y']) & 0x7FF))

            buffer[i] = value & 0xFF
            buffer[i+1] = (value >> 8) & 0xFF
            i += 2

        # EVT_ADDR_X - Single valid event, X coordinate and polarity ‘0010’
        value = EVT3_EVT_ADDR_X | (int(ev['x']) & 0x7FF) | ((int(ev['p']) & 0x01) << 11)

        buffer[i] = value & 0xFF
        buffer[i+1] = (value >> 8) & 0xFF
        i += 2


    return buffer[:i], last_lower12_ts, last_upper12_ts, last_y




@nb.njit
def parse_evt3_buffer(input_buffer: np.ndarray, events_buffer: np.ndarray, triggers_buffer: np.ndarray, last_ts_high_high: np.int64, last_ts_high: np.int64, last_ts_low: np.int64, last_y: np.int16, ts_initialized: bool):


    n_events = 0
    n_ext_triggers = 0
    i = 0


    if not ts_initialized:
        # Find the first EVT_TIME_HIGH
        while i < len(input_buffer):

            packet_type = input_buffer[i] & 0xF000
            if packet_type == EVT3_EVT_TIME_HIGH:
                ts_initialized = True
                break
            i += 1


    while i < len(input_buffer):
        packet_type = input_buffer[i] & 0xF000
        packet_value = input_buffer[i] & 0x0FFF
        # print(f"{value:04x}")



        # EVT_TIME_HIGH - Updates the higher 12-bit portion of the 24-bit time base ‘1000’
        if packet_type == EVT3_EVT_TIME_HIGH:
            if last_ts_high > packet_value:
                # Overflow
                # print("High Overflow detected")
                last_ts_high_high += 1

            last_ts_high = packet_value
            # print(f"EVT_TIME_HIGH {last_ts_high} - {packet_value} {packet_value << 12} {value:04x}")

        # EVT_TIME_LOW - Updates the lower 12-bit portion of the 24-bit time base ‘0110’
        elif packet_type == EVT3_EVT_TIME_LOW:
            # if last_ts_low > packet_value:
            #     # Overflow
            #     # print("Overflow detected")
            #     last_ts_high += 1
            last_ts_low = packet_value
            # print(f"EVT_TIME_LOW {last_ts_low} - {last_ts_high_high | last_ts_high | last_ts_low} {value:04x}")

        # EVT_ADDR_Y - Y coordinate, and system type (master/slave camera) ‘0000’
        elif packet_type == EVT3_EVT_ADDR_Y:
            last_y = packet_value & 0x7FF
            # print(f"EVT_ADDR_Y {last_y}")
        # EVT_ADDR_X - Single valid event, X coordinate and polarity ‘0010’
        elif packet_type == EVT3_EVT_ADDR_X:
            x = packet_value & 0x7FF
            p = 1 if (packet_value & 0x0800) else 0
            last_ts = (last_ts_high_high << 24) | (last_ts_high << 12) | last_ts_low

            # print(f"Event: {last_ts}, {x}, {last_y}, {p}")
            events_buffer[n_events]['t'] = last_ts
            events_buffer[n_events]['x'] = x
            events_buffer[n_events]['y'] = last_y
            events_buffer[n_events]['p'] = p

            n_events += 1
        elif packet_type == EVT3_VECT_BASE_X:
            
            last_ts = (last_ts_high_high << 24) | (last_ts_high << 12) | last_ts_low
            vect_base_x = packet_value & 0x7FF
            vect_base_p = (packet_value & 0x0800) >> 11

            if not i + 4 < len(input_buffer):
                break
                
            # Prevent buffer overflow crash
            if n_events + 32 > len(events_buffer):
                break

            value12_1 = input_buffer[i+1]
            if value12_1 & 0xF000 != EVT3_VECT_12:
                i += 2
                continue

            value12_2 = input_buffer[i+2]
            if value12_2 & 0xF000 != EVT3_VECT_12:
                i += 3
                continue

            value8_3 = input_buffer[i+3]
            if value8_3 & 0xF000 != EVT3_VECT_8:
                i += 4
                continue

            # Unroll and write directly to the buffer (C-style)
            for bit in range(12):
                if (value12_1 >> bit) & 0x01:
                    events_buffer[n_events]['t'] = last_ts
                    events_buffer[n_events]['x'] = vect_base_x + bit
                    events_buffer[n_events]['y'] = last_y
                    events_buffer[n_events]['p'] = vect_base_p
                    n_events += 1

            for bit in range(12):
                if (value12_2 >> bit) & 0x01:
                    events_buffer[n_events]['t'] = last_ts
                    events_buffer[n_events]['x'] = vect_base_x + 12 + bit
                    events_buffer[n_events]['y'] = last_y
                    events_buffer[n_events]['p'] = vect_base_p
                    n_events += 1

            for bit in range(8):
                if (value8_3 >> bit) & 0x01:
                    events_buffer[n_events]['t'] = last_ts
                    events_buffer[n_events]['x'] = vect_base_x + 24 + bit
                    events_buffer[n_events]['y'] = last_y
                    events_buffer[n_events]['p'] = vect_base_p
                    n_events += 1

            i += 3


        # elif packet_type == EVT3_VECT_12:
        #     print(f"VECT_12 {packet_value:04x}")
        # elif packet_type == EVT3_VECT_8:
        #     print(f"VECT_8 {packet_value:04x}")
        elif packet_type == EVT3_EXT_TRIGGER:
            # print(f"EXT_TRIGGER {packet_value:04x}")
            last_ts = (last_ts_high_high << 24) | (last_ts_high << 12) | last_ts_low
            p = 0x01 & packet_value

            # Channel ID bits 8..11
            id = (packet_value >> 8) & 0x0F

            triggers_buffer[n_ext_triggers]['t'] = last_ts
            triggers_buffer[n_ext_triggers]['p'] = p
            triggers_buffer[n_ext_triggers]['id'] = id

            n_ext_triggers += 1
        elif packet_type == EVT3_OTHERS:
            # print(f"OTHERS {packet_value:04x}")
            pass
        elif packet_type == EVT3_CONTINUED_4:
            # print(f"CONTINUED_4 {packet_value:04x}")
            pass
        elif packet_type == EVT3_CONTINUED_12:
            # print(f"CONTINUED_12 {packet_value:04x}")
            pass
        else:
            # print(f"Unknown value {value:04x}")
            pass
        i += 1

    return n_events, n_ext_triggers, last_ts_high_high, last_ts_high, last_ts_low, last_y, ts_initialized, i
    # else:
    #     return events[:n_events], triggers[:n_ext_triggers], last_ts, last_y, i


class EventDecoder_RAW(EventDecoder):
    '''
    Class for reading RAW files from Prophesee cameras

    Parameters
    ----------
    file : str
        Path to the RAW file
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
        super().__init__(readable, chunk_size)

        # EVT specific variables
        self._last_ts_high_high = 0
        self._last_ts_high = 0
        self._last_ts_low = 0
        self._ts_initialized = False
        self._last_y = 0
        self._format = None

        self._events_buffer = np.empty(self._chunk_size, dtype=Event_dtype)
        self._events_buffer_len = 0
        self._triggers_buffer = np.empty(self._chunk_size, dtype=Trigger_dtype)
        self._triggers_buffer_len = 0

    def init(self):

        # Try to find the end of the header
        is_header = True

        self._header = {
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
            pos = self._fd.tell()
            line = self._fd.readline()
            if line.startswith(b"% end"):
                is_header = False
                break
            if not line.startswith(b"%"):
                is_header = False
                self._fd.seek(pos)
                break

            split = line.decode('utf-8').strip().split(" ")
            key = split[1].lower()
            value = " ".join(split[2:])

            try:
                if key in self._header.keys():
                    if key == "date":
                        value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                    elif key == "height" or key == "width" or key == "system_id":
                        value = int(value)

                    self._header[key] = value
                else:
                    print(f"Unknown key {key} in header")
                    print(f"Supported keys are: {list(self._header.keys())}")
                    print(f"Line: {line}")
            except ValueError:
                print(f"Error parsing line: {line}")


        # Check Format, height and width:
        if self._header["format"] is not None:
            split: List[str] = self._header["format"].split(";")
            for s in split:
                if s.startswith("height"):
                    self._height = int(s.split("=")[1])
                elif s.startswith("width"):
                    self._width = int(s.split("=")[1])
                else:
                    s = s.lower().replace(".", "")
                    if s in EventDecoder_RAW.FORMATS.keys():
                        self._format = s
                    else:
                        print(f"Unknown format {s}, supported formats are {list(EventDecoder_RAW.FORMATS.keys())}")

        if self._header["geometry"] is not None:
            split = self._header["geometry"].split("x")
            if len(split) != 2:
                raise ValueError(f"Invalid geometry {self._header['geometry']}")

            if self._width is not None and self._height is not None:
                if self._width != int(split[0]) or self._height != int(split[1]):
                    print(f"Warning: Geometry in header {self._header['geometry']} does not match width/height")
                    print(f"Setting width/height to {split[0]}x{split[1]}")
            self._width = int(split[0])
            self._height = int(split[1])

        if self._header['evt'] is not None:

            if self._header['evt'] in EventDecoder_RAW.EVT_FORMATS.keys():
                format = EventDecoder_RAW.EVT_FORMATS[self._header['evt']]

                if self._format is not None and self._format != format:
                    print(f"Warning: evt in header {self._header['evt']} does not match evt")
                    print(f"Setting evt to {format}")
                    self._format = format


            else:
                print(f"Unknown evt version {self._header['evt']}")
                print(f"Supported evt versions are: {list(EventDecoder_RAW.EVT_FORMATS.keys())}")


        if self._format is None:
            print(f"Error: Format not found in header, trying to use EVT3")
            self._format = "evt3"

        if self._width is None or self._height is None:
            if self._header['sensor_name'] == "IMX636":
                self._width = 1280
                self._height = 720

        if self._width is None or self._width < 0 or self._width > 2048:
            print(f"Error: Valid Width not found in header, setting to 2048")
            self._width = 2048
        if self._height is None or self._height < 0 or self._height > 2048:
            print(f"Error: Valid Height not found in header, setting to 2048")
            self._height = 2048

        self._is_initialized = True


    def read_chunk(self, delta_t_hint:int = None, n_events_hint:int = None) -> np.ndarray:
        if not self._is_initialized:
            self.init()

        events, triggers = self.__read_and_parse_buffer()
        return events


    def __read_and_parse_buffer(self):
        # print(f"Reading buffer of size {self.buffer_size}")
        assert self._fd is not None

        if self._format == "evt3":

            # Read the buffer as uint16
            input_buffer = np.fromfile(self._fd, dtype=np.uint16, count=self._chunk_size)

            # We have reached the end of the file, but not nessarily the end of the buffer
            if len(input_buffer) < self._chunk_size:
                self.eof = True

            if len(input_buffer) == 0:
                return np.array([], dtype=Event_dtype), np.array([], dtype=Trigger_dtype)

            n_events, n_triggers, self._last_ts_high_high, self._last_ts_high, self._last_ts_low, self._last_y, self._ts_initialized, msg_processed = parse_evt3_buffer(
                    input_buffer,
                    self._events_buffer,
                    self._triggers_buffer,
                    self._last_ts_high_high,
                    self._last_ts_high,
                    self._last_ts_low,
                    self._last_y,
                    self._ts_initialized)

            # self.events_buffer_len += n_events
            # self.triggers_buffer_len += n_triggers
            if msg_processed < len(input_buffer):
                # This happens if We are in the middle of a Vect message and we need to fetch more data
                missed_bytes = 2 * (len(input_buffer) - msg_processed)
                # print(f"Missed {missed_bytes} bytes, seeking back")
                self._fd.seek(-missed_bytes, 1)



            del input_buffer
        elif self._format == "evt21":
            raise NotImplementedError("EVT2.1 not implemented")
        elif self._format == "evt2":
            raise NotImplementedError("EVT2 not implemented")
        else:
            raise ValueError(f"Unsupported format {self._format}. Supported formats are {list(EventDecoder_RAW.FORMATS.keys())}")
        return self._events_buffer[:n_events], self._triggers_buffer[:n_triggers]


    def reset(self):
        assert self._fd is not None
        self._fd.seek(0)
        self._is_initialized = False
        self._eof = False




class EventEncoder_RAW(EventEncoder):
    '''
    Class for writing RAW files from Prophesee cameras

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
        if format not in EventEncoder_RAW.FORMATS.keys():
            raise ValueError(f"Unsupported format {format}. Supported formats are {list(EventEncoder_RAW.FORMATS.keys())}")
        self._format = format

        self._system_id = 49

        self._last_upper12_ts = -1
        self._last_lower12_ts = -1
        self._last_y  = -1

        self._serial_number = serial

        self._formatted_datetime = self._dt.strftime("%Y-%m-%d %H:%M:%S")

    def init(self):
        if self._is_initialized:
            return

        self._fd.write(
f"""% camera_integrator_name Prophesee
% date {self._formatted_datetime}
% {self.FORMATS[self._format]}
% format {self._format};height={self._height};width={self._width}
% generation 4.2
% geometry {self._width}x{self._height}
% integrator_name Prophesee
% plugin_integrator_name Prophesee
% plugin_name hal_plugin_prophesee
% sensor_generation 4.2
% serial_number {self._serial_number}
% system_ID {self._system_id}
% end
""".encode('utf-8'))
        self._is_initialized = True


    def write(self, events: np.ndarray) -> int:
        assert self._fd is not None

        if not self._is_initialized:
            self.init()

        if self._format == "evt3":
            buffer, self._last_lower12_ts, self._last_upper12_ts, self._last_y = get_raw_evt3_buffer(
                events,
                self._last_lower12_ts,
                self._last_upper12_ts,
                self._last_y)
        elif self._format == "evt21":
            raise NotImplementedError("EVT2.1 not implemented")
        elif self._format == "evt2":
            raise NotImplementedError("EVT2 not implemented")
        else:
            raise NotImplementedError(f"Unsupported format {self._format}. Supported formats are {list(EventEncoder_RAW.FORMATS.keys())}")

        self._n_written_events += len(events)

        buffer.tofile(self._fd)

        return len(events)
