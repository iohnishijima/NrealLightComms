import zmq

class EyeTrackingReceiver:
    def __init__(self, local_ip, remote_ip, port, use_remote, shared_data, popup_data):
        self.lip = local_ip
        self.rip = remote_ip
        self.p = port
        self.status = use_remote
        self.BB = 0.0
        self.BE = 0.0
        self.eye_detected = True
        self.shared_data = shared_data
        self.popup_data = popup_data
        self.blink_begin = False
        self.blink_end = False
        self.count_rest = 0

        # Initialize ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)

        # Connect to appropriate IP
        if self.status:
            self.socket.connect(f"tcp://{self.rip}:{self.p}")
        else:
            self.socket.connect(f"tcp://{self.lip}:{self.p}")
        print(f"Attempting to connect to {'remote' if self.status else 'local'} server at {'tcp://' + self.rip + ':' + str(self.p) if self.status else 'tcp://' + self.lip + ':' + str(self.p)}")

    def receive_data(self):
        while True:
            try:
                text = self.socket.recv_string(zmq.DONTWAIT)
                if text:
                    self.parse_data(text)
            except zmq.error.Again:
                pass

            except KeyboardInterrupt:
                break

    
    def parse_data(self, data_text):
        data = data_text.split(";")

        try:
            server_data = {
                'ID' : float(data[0]),
                'Timestamp' : float(data[1]),
                'GazeX' : float(data[2]),
                'GazeY' : float(data[3]),
                'RScore' : float(data[9]),
                'LScore' : float(data[10]),
                'eyeEvent': str(data[20])
            }
            
            if data[20] == " NA":
                if self.eye_detected:
                    print("eyes are not detected")
                    self.eye_detected = False
            else:
                if not self.eye_detected:
                    print("eyes are detected, analyze start")
                    self.eye_detected = True
                if server_data['eyeEvent'] == "BB":
                    self.blink_begin = True
                elif server_data['eyeEvent'] == "BB":
                    self.blink_begin = False
                    self.blink_end = True

                if self.blink_begin == True:
                    self.shared_data['ID'].value = server_data["ID"]
                    self.shared_data['Timestamp'].value = server_data["Timestamp"]
                    self.shared_data['GazeX'].value = None
                    self.shared_data['GazeY'].value = None
                    self.shared_data['RScore'].value = server_data["RScore"]
                    self.shared_data['LScore'].value = server_data["LScore"]
                    self.shared_data['eyeEvent'].value = server_data["eyeEvent"]
                    self.popup_data['GazeX'].value = None
                    self.popup_data['GazeY'].value = None
                elif self.blink_end == True and self.count_rest < 10:
                    self.shared_data['ID'].value = server_data["ID"]
                    self.shared_data['Timestamp'].value = server_data["Timestamp"]
                    self.shared_data['GazeX'].value = None
                    self.shared_data['GazeY'].value = None
                    self.shared_data['RScore'].value = server_data["RScore"]
                    self.shared_data['LScore'].value = server_data["LScore"]
                    self.shared_data['eyeEvent'].value = server_data["eyeEvent"]
                    self.popup_data['GazeX'].value = None
                    self.popup_data['GazeY'].value = None
                    self.count_rest += 1
                elif self.count_rest >= 10:
                    self.count_rest = 0
                    self.blink_end = False
                else:
                    self.shared_data['ID'].value = server_data["ID"]
                    self.shared_data['Timestamp'].value = server_data["Timestamp"]
                    self.shared_data['GazeX'].value = server_data["GazeX"]
                    self.shared_data['GazeY'].value = server_data["GazeY"]
                    self.shared_data['RScore'].value = server_data["RScore"]
                    self.shared_data['LScore'].value = server_data["LScore"]
                    self.shared_data['eyeEvent'].value = server_data["eyeEvent"]
                    self.popup_data['GazeX'].value = server_data["GazeX"]
                    self.popup_data['GazeY'].value = server_data["GazeY"]
                print(F"{self.shared_data['GazeX'].value}, {self.shared_data['GazeY'].value}")
                # print(server_data)
                # print(data)
                # self.analyze_data(server_data)

        except Exception as e:
            print(f"Error parsing data: {e}")


    def analyze_data(self, server_data):
        if server_data["eyeEvent"] == ' BB':
            self.BB = server_data["Timestamp"]
        elif server_data["eyeEvent"] == ' BE':
            self.BE = server_data["Timestamp"]
            print(f"blink was {self.BE - self.BB} ms")