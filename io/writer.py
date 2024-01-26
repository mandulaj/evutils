from datetime import datetime


import numpy as np


 


class EventWriter():
    def __init__(self, file, width=1280, height=720, dt: datetime = None):

        self.file = file
        self.fd = None 
        self.width = width
        self.height = height

        

        if dt is None:
            self.dt = datetime.now()
        else:
            self.dt = dt


    def init(self):
        raise NotImplementedError
    
    def write(self, event: np.ndarray):
        raise NotImplementedError
    
    def __enter__(self):
        return self
    
    def close(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

 
class EventWriter_RAW(EventWriter):
    def __init__(self, file, width=1280, height=720, dt=None, serial="00000000"):
        super().__init__(file, width, height, dt)

        self.was_initialized = False

        self.last_upper12_ts = 0
        self.last_lower12_ts = 0
        self.last_y  = 0 

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


        for ev in events:
            upper12_ts = (int(ev['t']) & 0x0FFF000) >> 12
            lower12_ts = int(ev['t']) & 0x00000FFF

            if upper12_ts != self.last_upper12_ts:
                # write new upper ts2000
                self.last_upper12_ts = upper12_ts
                value = 0x8000 | upper12_ts
                self.fd.write(value.to_bytes(2, byteorder='little'))
            
            if lower12_ts != self.last_lower12_ts:
                # write new lower ts
                self.last_lower12_ts = lower12_ts
                value = 0x6000 | lower12_ts
                self.fd.write(value.to_bytes(2, byteorder='little'))
                                
            if self.last_y != ev['y']:
                value = (0x0000 | int(ev['y']))
                # write new y
                self.fd.write(value.to_bytes(2, byteorder='little'))
                self.last_y = ev['y']
            # write x and p
            value = 0x2000 | int(ev['x']) | int(ev['p'] << 11)
            self.fd.write(value.to_bytes(2, byteorder='little'))


class EventWriter_CSV(EventWriter):
    def __init__(self, file, width=1280, height=720):
        super().__init__(file, width, height)

    def init(self):
        self.fd = open(self.file, "w")
        self.fd.write("t, x, y, p\n")

    def write(self, events: np.ndarray):
        for ev in events:
            self.fd.write(f"{ev['t']}, {ev['x']}, {ev['y']}, {ev['p']}\n")

    def close(self):
        self.fd.close()

class EventWriter_HDF5(EventWriter):
    def __init__(self, file, width=1280, height=720):
        super().__init__(file, width, height)

class EventWriter_Bin(EventWriter):
    def __init__(self, file, width=1280, height=720):
        super().__init__(file, width, height)


class EventWriter_Any(EventWriter):

    def __init__(self, file, width=1280, height=720):
        super().__init__(file, width, height)



