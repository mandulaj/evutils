

import numpy as np
import numba as nb

import datetime as dt

from ._writer import EventWriter
from ._reader import EventReader

from ..types import Events, Triggers

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




# @nb.njit
def parse_evt3_buffer(buffer: bytes, last_ts_high_high: np.int64, last_ts_high: np.int64, last_ts_low: np.int64, last_y: np.int16):

    events = np.zeros(len(buffer) // 2, dtype=Events)
    triggers = np.zeros(len(buffer) // 2, dtype=Triggers)
    n_events = 0
    n_ext_triggers = 0
    i = 0

    if last_ts_low == -1:
        # Find the first EVT_TIME_HIGH
        while i + 2 <= len(buffer):
            packet_type = (int(buffer[i+1]) << 8) | int(buffer[i]) & 0xF000
            if packet_type != EVT3_EVT_TIME_HIGH:
                i += 2
            else:
                break
            


    while i + 2 <= len(buffer):
        value = (int(buffer[i+1]) << 8) | int(buffer[i])
        packet_type = value & 0xF000
        packet_value = value & 0x0FFF
        # print(f"{value:04x}")



        # EVT_TIME_HIGH - Updates the higher 12-bit portion of the 24-bit time base ‘1000’
        if packet_type == EVT3_EVT_TIME_HIGH:
            new_ts_high = (packet_value << 12)

            if last_ts_high > new_ts_high:
                # Overflow
                # print("High Overflow detected")
                last_ts_high_high += 0x100000000

            last_ts_high = new_ts_high
            # print(f"EVT_TIME_HIGH {last_ts_high} - {packet_value} {packet_value << 12} {value:04x}")
        
        # EVT_TIME_LOW - Updates the lower 12-bit portion of the 24-bit time base ‘0110’
        elif packet_type == EVT3_EVT_TIME_LOW:
            if last_ts_low > packet_value:
                # Overflow
                # print("Overflow detected")
                last_ts_high += 0x1000
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
            last_ts = last_ts_high_high | last_ts_high | last_ts_low
 
            # print(f"Event: {last_ts}, {x}, {last_y}, {p}")
            events[n_events]['t'] = last_ts 
            events[n_events]['x'] = x 
            events[n_events]['y'] = last_y 
            events[n_events]['p'] = p 
            
            n_events += 1
        elif packet_type == EVT3_VECT_BASE_X:
            last_ts = last_ts_high_high | last_ts_high | last_ts_low
            vect_base_x = packet_value & 0x7FF
            vect_base_p = (packet_value & 0x0800) >> 11

            
            if i + 8 > len(buffer):
                # print("Warning: Buffer is too small for VECT_BASE_X")
                # buffer_too_small = True
                break

            # print(f"VECT_BASE_X {packet_value:04x}")

            # print("VECT_BASE_X", vect_base_x, vect_base_p, last_ts)


            vect_events = np.empty(32, dtype=Events)
            vect_events['p'][:] = vect_base_p
            vect_events['y'][:] = last_y
            vect_events['t'][:] = last_ts
            vect_events['x'][:] = np.arange(32) + vect_base_x

            valid = np.zeros(32, dtype=np.uint8)
            
            i += 2
            value12_1 = (int(buffer[i+1]) << 8) | int(buffer[i])
            if value12_1 & 0xF000 != EVT3_VECT_12:
                # print(f"Warning: Expected VECT_12, got {value12_1:04x}")
                continue

            for pos, bit in enumerate(range(0, 12)):
                valid[bit] = (value12_1 >> pos) & 0x01
            
            i += 2
            value12_2 = (int(buffer[i+1]) << 8) | int(buffer[i])
            if value12_2 & 0xF000 != EVT3_VECT_12:
                # print(f"Warning: Expected VECT_12, got {value12_1:04x}")
                continue

            for pos, bit in enumerate(range(12, 24)):
                valid[bit] = (value12_2 >> pos) & 0x01

            i += 2
            value8_3 = (int(buffer[i+1]) << 8) | int(buffer[i])
            if value8_3 & 0xF000 != EVT3_VECT_8:
                # print(f"Warning: Expected VECT_8, got {value12_1:04x}")
                continue

            for pos, bit in enumerate(range(24, 32)):
                valid[bit] = (value8_3 >> pos) & 0x01

            # print(vect_events)
            vect_events = vect_events[valid == 1]

            # print(vect_events)
            events[n_events:n_events+len(vect_events)] = vect_events
            n_events += len(vect_events)
            # exit()


        # elif packet_type == EVT3_VECT_12:
        #     print(f"VECT_12 {packet_value:04x}")
        # elif packet_type == EVT3_VECT_8:
        #     print(f"VECT_8 {packet_value:04x}")
        elif packet_type == EVT3_EXT_TRIGGER:
            # print(f"EXT_TRIGGER {packet_value:04x}")
            last_ts = last_ts_high_high | last_ts_high | last_ts_low
            p = 0x01 & packet_value
            
            # Channel ID bits 8..11
            id = (packet_value >> 8) & 0x0F

            triggers[n_ext_triggers]['t'] = last_ts
            triggers[n_ext_triggers]['p'] = p
            triggers[n_ext_triggers]['id'] = id
            
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
        i += 2
    
    return events[:n_events], triggers[:n_ext_triggers], last_ts_high_high, last_ts_high, last_ts_low, last_y, i
    # else:
    #     return events[:n_events], triggers[:n_ext_triggers], last_ts, last_y, i
        

class EventReader_RAW(EventReader): 
    MAX_EVENTS_READ = 1e12
    MAX_DELTA_T = 1e12
    def __init__(self, file,  delta_t=10000, n_events=10000, mode="delta_t", buffer_size=4005):
        super().__init__(file, delta_t, n_events, mode)

        self.buffer_position = -1
        self.buffer_size = buffer_size
        self.buffer = []
        
        self.last_ts_high_high = 0
        self.last_ts_high = 0
        self.last_ts_low = np.int64(-1)
        self.last_y = np.int16(0)
        self.format = None

    def init(self):
        self.fd = open(self.file, "rb")


        # Try to find the end of the header
        is_header = True

        self.header = {
            "date": dt.datetime.now(),
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
                        value = dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
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
            split = self.header["format"].split(";")
            for s in split:
                if s.startswith("height"):
                    self.height = int(s.split("=")[1])
                elif s.startswith("width"):
                    self.width = int(s.split("=")[1])
                else:
                    if s in ["EVT2", "EVT2.1", "EVT3"]:
                        self.format = s
                    else:
                        print(f"Unknown format {s}")
                        print(f"Supported formats are: EVT2, EVT2.1, EVT3")
        
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

            FORMATS = {"3.0": "EVT3", "2.1": "EVT2.1", "2.0": "EVT2"}
            if self.header['evt'] in FORMATS.keys():
                format = FORMATS[self.header['evt']]

                if self.format is not None and self.format != format:
                    print(f"Warning: evt in header {self.header['evt']} does not match evt")
                    print(f"Setting evt to {format}")
                    self.format = format
                

            else:
                print(f"Unknown evt version {self.header['evt']}")
                print(f"Supported evt versions are: {list(FORMATS.keys())}")
        

        if self.format is None:
            print(f"Error: Format not found in header, trying to use EVT3")
            self.format = "EVT3"

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

    def read(self, delta_t=None, n_events=None) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_initialized:
            self.init()


        if delta_t is None:
            delta_t = self.delta_t
        if n_events is None:
            n_events = self.n_events


        

        if self.mode == "delta_t":
            n_events = EventReader_RAW.MAX_EVENTS_READ
        if self.mode == "n_events":
            delta_t = EventReader_RAW.MAX_DELTA_T
        
    
        
        all_events = np.zeros(0, dtype=Events)
        all_triggers = np.zeros(0, dtype=Triggers)



        while len(all_events) < n_events:
            
            if self.buffer_position >= len(self.buffer) or self.buffer_position == -1:
                self.buffer = self.fd.read(self.buffer_size)
                self.buffer_position = 0

                if len(self.buffer) == 0:
                    self.eof = True
                    break

            if self.format == "EVT3":
                print("Current Time:", self.last_ts_high_high | self.last_ts_high | self.last_ts_low)
                # Parse EVT3 format

                # print(self.buffer)
                events, triggers, self.last_ts_high_high, self.last_ts_high, self.last_ts_low, self.last_y, bytes_processed = parse_evt3_buffer(self.buffer[self.buffer_position:], self.last_ts_high_high, self.last_ts_high, self.last_ts_low, self.last_y)
                
                self.buffer_position += bytes_processed
                if self.buffer_position < len(self.buffer):

                    new_buffer = self.fd.read(self.buffer_size)
                    if len(new_buffer) == 0:
                        self.eof = True
                        
                    self.buffer = self.buffer[self.buffer_position + bytes_processed:] + new_buffer
                    self.buffer_position = 0


            all_events = np.concatenate((all_events, events))
            all_triggers = np.concatenate((all_triggers, triggers))

            if len(all_events) > 1 and all_events[-1]['t'] - all_events[0]['t'] > delta_t:
                break 

            
   
        return all_events, all_triggers






        return np.array([], dtype=Events), not self.eof

        

        if self.buffer_position > len(self.buffer) or self.buffer_position == -1:
            self.buffer = self.fd.read(2 * self.buffer_size)
            self.buffer_position = 0

        if len(self.buffer) == 0:
            self.eof = True
            return np.array([], dtype=Events), False

        print(self.buffer)


    
    def close(self):
        if self.is_initialized:
            self.fd.close()



 
class EventWriter_RAW(EventWriter):
    FORMATS = {"EVT3": "evt 3.0", "EVT2.1": "evt 2.1", "EVT2": "evt 2.1"}
    def __init__(self, file, width=1280, height=720, dt=None, serial="00000000", format="EVT3"):
        super().__init__(file, width, height, dt)

        if format not in self.FORMATS.keys():
            raise ValueError(f"Unsupported format {format}. Supported formats are {list(EventWriter_RAW.FORMATS.keys())}")
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
        
        self.fd = open(self.file, "wb")

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
        
    def close(self):
        if self.is_initialized:
            self.fd.close()

    def write(self, events: np.ndarray):

        if not self.is_initialized:
            self.init()

        if self.format == "EVT3":
            buffer, self.last_lower12_ts, self.last_upper12_ts, self.last_y = get_raw_evt3_buffer(
                events, 
                self.last_lower12_ts, 
                self.last_upper12_ts, 
                self.last_y)
        else:
            raise NotImplementedError(f"Unsupported format {self.format}. Supported formats are {list(EventWriter_RAW.FORMATS.keys())}")

        self.n_written_events += len(events)

        buffer.tofile(self.fd)