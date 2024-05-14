import zmq
import numpy as np
import cv2
import time

class SceneImageReceiver:
    def __init__(self, local_ip, remote_ip, port, use_remote, image_buffer_scene):
        self.lip = local_ip
        self.rip = remote_ip
        self.p = port
        self.status = use_remote
        self.image_buffer_scene = image_buffer_scene

        # Initialize ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.RCVTIMEO = 300

        # Connect to appropriate IP
        if self.status:
            self.socket.connect(f"tcp://{self.rip}:{self.p}")
        else:
            self.socket.connect(f"tcp://{self.lip}:{self.p}")
        print(f"Attempting to connect to {'remote' if self.status else 'local'} server at {'tcp://' + self.rip + ':' + str(self.p) if self.status else 'tcp://' + self.lip + ':' + str(self.p)}")

    def receive_data(self):
        pre_data = None
        while True:
            try:
                message = self.socket.recv()
                frame_number = int.from_bytes(message[:4], byteorder='big')
                image_data = message[4:]
                arr = np.frombuffer(image_data, dtype=np.uint8)
                img = cv2.imdecode(arr, flags=cv2.IMREAD_COLOR)
                self.image_buffer_scene[:] = img
                pre_data = img
            except zmq.error.Again:
                if pre_data is not None:
                    self.image_buffer_scene[:] = pre_data
                pass

            except KeyboardInterrupt:
                break