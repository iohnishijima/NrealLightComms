import hid
import time
from crc import crc32_custom 

class NrealLight:
    MCU_VID = 0x0486
    MCU_PID = 0x573c

    def __init__(self):
        self.device = None
        for device_dict in hid.enumerate():
            if device_dict['vendor_id'] == self.MCU_VID and device_dict['product_id'] == self.MCU_PID:
                self.device = hid.device()
                self.device.open_path(device_dict['path'])
        if self.device is None:
            raise Exception("Nreal Light device not found.")
        else:
            print("Device opened successfully.")
        
    def send_command(self, command_type, cmd, cmd_data='x'):
        timestamp = '00000000'  
        crc_string = f'\x02:{command_type}:{cmd}:{cmd_data}:{timestamp}:'
        # Ensure the crc_string is converted to bytes before passing to crc32_custom
        crc_bytes = crc_string.encode()
        crc = crc32_custom(crc_bytes)
        command = f'{crc_string}{crc:08x}:\x03'
        print(f"Sending command: {command}")
        # Convert the entire command to bytes before sending
        command_bytes = [ord(c) for c in command]
        self.device.write(command_bytes)
        time.sleep(0.1)
        response = self.device.read(64, timeout_ms=2000)
        return response
