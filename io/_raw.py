

import numpy as np
import numba as nb


from ._writer import EventWriter
from ._reader import EventReader



@nb.njit
def get_raw_buffer(events: np.ndarray, last_lower12_ts: int, last_upper12_ts: int, last_y: int):
    buffer = np.zeros(len(events) * 8, dtype=np.uint8)

    length = 0

    for ev in events:
        upper12_ts = (int(ev['t']) & 0x0FFF000) >> 12
        lower12_ts = int(ev['t']) & 0x00000FFF

        if upper12_ts != last_upper12_ts:
            # write new upper ts2000
            last_upper12_ts = upper12_ts
            value = 0x8000 | (upper12_ts & 0xFFF)
            buffer[length+1] = (value >> 8) & 0xFF
            buffer[length] = value & 0xFF
            length += 2
        
        if lower12_ts != last_lower12_ts:
            # write new lower ts
            last_lower12_ts = lower12_ts 
            value = 0x6000 | (lower12_ts & 0xFFF)
            buffer[length+1] = (value >> 8) & 0xFF
            buffer[length] = value & 0xFF
            length += 2
                            
        if last_y != ev['y']:
            value = (0x0000 | (int(ev['y']) & 0xFFF))
            # write new y
            buffer[length+1] = (value >> 8) & 0xFF
            buffer[length] = value & 0xFF
            length += 2
            last_y = ev['y']
        # write x and p
        value = 0x2000 | (int(ev['x']) & 0xFFF) | ((int(ev['p']) & 0x01) << 11)
        buffer[length+1] = (value >> 8) & 0xFF
        buffer[length] = value & 0xFF
        length += 2


    return buffer[:length], last_lower12_ts, last_upper12_ts, last_y



        

class EventReader_RAW(EventReader):
    def __init__(self, file):
        super().__init__(file)



 
class EventWriter_RAW(EventWriter):
    def __init__(self, file, width=1280, height=720, dt=None, serial="00000000"):
        super().__init__(file, width, height, dt)

        self.was_initialized = False

        self.last_upper12_ts = -1
        self.last_lower12_ts = -1
        self.last_y  = -1

        self.serial_number = serial
       
        self.formatted_datetime = self.dt.strftime("%Y-%m-%d %H:%M:%S")

        self.init()


    def init(self):
        if self.was_initialized:
            return
        
        self.fd = open(self.file, "wb")

        self.fd.write(
f"""% camera_integrator_name Prophesee
% date {self.formatted_datetime}
% evt 3.0
% format EVT3;height={self.height};width={self.width}
% generation 4.2
% geometry {self.width}x{self.height}
% integrator_name Prophesee
% plugin_integrator_name Prophesee
% plugin_name hal_plugin_prophesee
% sensor_generation 4.2
% serial_number {self.serial_number}
% system_ID 49
% end
""".encode('utf-8'))
        self.was_initialized = True
        
    def close(self):
        self.fd.close()

    def write(self, events: np.ndarray):

        buffer, self.last_lower12_ts, self.last_upper12_ts, self.last_y = get_raw_buffer(
            events, 
            self.last_lower12_ts, 
            self.last_upper12_ts, 
            self.last_y)
        
        buffer.tofile(self.fd)