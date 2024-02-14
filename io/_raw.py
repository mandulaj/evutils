

import numpy as np
import numba as nb


from ._writer import EventWriter
from ._reader import EventReader



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
            value = 0x8000 | (upper12_ts & 0xFFF)
            
            buffer[i] = value & 0xFF
            buffer[i+1] = (value >> 8) & 0xFF
            i += 2
        
        # EVT_TIME_LOW - Updates the lower 12-bit portion of the 24-bit time base ‘0110’
        if lower12_ts != last_lower12_ts:
            last_lower12_ts = lower12_ts 
            value = 0x6000 | (lower12_ts & 0xFFF)
            
            buffer[i] = value & 0xFF
            buffer[i+1] = (value >> 8) & 0xFF
            i += 2

        # EVT_ADDR_Y - Y coordinate, and system type (master/slave camera) ‘0000’
        if last_y != ev['y']:
            last_y = ev['y']
            value = (0x0000 | master_slave | (int(ev['y']) & 0x7FF))
            
            buffer[i] = value & 0xFF
            buffer[i+1] = (value >> 8) & 0xFF
            i += 2
        
        # EVT_ADDR_X - Single valid event, X coordinate and polarity ‘0010’
        value = 0x2000 | (int(ev['x']) & 0x7FF) | ((int(ev['p']) & 0x01) << 11)
        
        buffer[i] = value & 0xFF
        buffer[i+1] = (value >> 8) & 0xFF
        i += 2


    return buffer[:i], last_lower12_ts, last_upper12_ts, last_y



        

class EventReader_RAW(EventReader):
    def __init__(self, file):
        super().__init__(file)



 
class EventWriter_RAW(EventWriter):
    FORMATS = {"EVT3": "evt 3.0", "EVT2.1": "evt 2.1", "EVT2": "evt 2.1"}
    def __init__(self, file, width=1280, height=720, dt=None, serial="00000000", format="EVT3"):
        super().__init__(file, width, height, dt)

        if format not in self.FORMATS.keys():
            raise ValueError(f"Unsupported format {format}. Supported formats are {list(EventWriter_RAW.FORMATS.keys())}")
        self.format = format
        self.system_id = 49

        self.was_initialized = False

        self.last_upper12_ts = -1
        self.last_lower12_ts = -1
        self.last_y  = -1

        self.serial_number = serial
       
        self.formatted_datetime = self.dt.strftime("%Y-%m-%d %H:%M:%S")

    def init(self):
        if self.was_initialized:
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
        self.was_initialized = True
        
    def close(self):
        if self.was_initialized:
            self.fd.close()

    def write(self, events: np.ndarray):

        if not self.was_initialized:
            self.init()

        if self.format == "EVT3":
            buffer, self.last_lower12_ts, self.last_upper12_ts, self.last_y = get_raw_evt3_buffer(
                events, 
                self.last_lower12_ts, 
                self.last_upper12_ts, 
                self.last_y)
        else:
            raise NotImplementedError(f"Unsupported format {self.format}. Supported formats are {list(EventWriter_RAW.FORMATS.keys())}")

        buffer.tofile(self.fd)