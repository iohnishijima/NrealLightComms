import argparse
import multiprocessing
from EyeTrackingReceiver import EyeTrackingReceiver
from ObjectDetectionProcessor import ObjectDetectionProcessor
from TextToSpeech import TextToSpeech
from PopupWindowConfirmation import Confirmation
from multiprocessing import shared_memory
from SceneImageReceiver import SceneImageReceiver
import numpy as np
# import signal
import sys

def object_detection_process(shared_data, watch_gaze, popup_data, image_buffer,flag_data, image_buffer_scene):
    print("enter object_detection_process")
    display_runner = ObjectDetectionProcessor(shared_data, watch_gaze, popup_data, image_buffer,flag_data, image_buffer_scene)
    display_runner.display()

# Eye Tracking Data Reception Setup
def eye_tracking_process(local_ip, remote_ip, port, use_remote, shared_data, popup_data):
    print("enter eye_tracking_process")
    receiver = EyeTrackingReceiver(local_ip, remote_ip, port, use_remote, shared_data, popup_data)
    receiver.receive_data()
    
def text_to_speech(watch_gaze, popup_data):
    print("enter eye_tracking_process")
    speaker = TextToSpeech(watch_gaze, popup_data)
    speaker.speech()
    
def popup_windiow(popup_data, image_buffer, flag_data):
    print("enter popup window")
    window = Confirmation(popup_data, image_buffer, flag_data)
    window.main_loop()
    
def scene_image(local_ip, remote_ip, port, use_remote, image_buffer_scene):
    print("enter scene_image")
    window = SceneImageReceiver(local_ip, remote_ip, port, use_remote, image_buffer_scene)
    window.receive_data()
    
def signal_handler(sig, frame, shm, shm_scene):
    print(f'Get {sig} Signal, finishing process')
    shm.close()
    shm_scene.close()
    shm.unlink()
    shm_scene.unlink()
    sys.exit(0)

def main():
    # signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, shm, shm_scene))
    # signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, shm, shm_scene))
    manager = multiprocessing.Manager()
    shared_data = {
        'ID': manager.Value('d', 0.0),
        'Timestamp': manager.Value('d', 0.0),
        'GazeX': manager.Value('d', 0.0),
        'GazeY': manager.Value('d', 0.0),
        'RScore': manager.Value('d', 0.0),
        'LScore': manager.Value('d', 0.0),
        'eyeEvent': manager.Value('u', u'')
    }
    
    gaze_manager = multiprocessing.Manager()
    watch_gaze = {
        'is_watching': gaze_manager.Value('b', False),
        'object_name': gaze_manager.Value('u', ''),
        'timestamp': gaze_manager.Value('d', 0.0)
    }
    
    popup_manager = multiprocessing.Manager()
    popup_data = {
        'Timestamp': popup_manager.Value('d', 0.0),
        'GazeX': popup_manager.Value('d', 0.0),
        'GazeY': popup_manager.Value('d', 0.0),
        'Confirmation': popup_manager.Value('b', False),
        'AddAllData': popup_manager.Value('b', False),
        'X': popup_manager.Value('d', 0.0),
        'Y': popup_manager.Value('d', 0.0),
        'W': popup_manager.Value('d', 0.0),
        'H': popup_manager.Value('d', 0.0),
    }
    
    flag_manager = multiprocessing.Manager()
    flag_data = {
        'glasses': flag_manager.Value('u', 'STONE'), # Nreal, STONE
    }
    
    max_image_size = 1920 * 1080 * 3  # width * height * channels (BGR)
    shm = shared_memory.SharedMemory(create=True, size=max_image_size)
    image_buffer = np.ndarray((1080, 1920, 3), dtype=np.uint8, buffer=shm.buf)
    
    if flag_data['glasses'].value == "STONE":
        max_image_size_scene = 640 * 480 * 3  # width * height * channels (BGR)
        shared_array = multiprocessing.Array('B', max_image_size_scene)
        img_array = np.frombuffer(shared_array.get_obj(), dtype=np.uint8).reshape((480, 640, 3))
        # shm_scene = shared_memory.SharedMemory(create=True, size=max_image_size_scene)
        # image_buffer_scene = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm_scene.buf)
    
    parser = argparse.ArgumentParser(description="IP addresses, port, and remote status for eye tracking data reception.")
    parser.add_argument("--local_ip", type=str, default="127.0.0.1", help="Local IP address")
    parser.add_argument("--remote_ip", type=str, default="192.168.1.117", help="Remote IP address")
    parser.add_argument("--port", type=int, default=3428, help="Port number")
    parser.add_argument("--use_remote", type=bool, default=False, help="Use remote IP? True/False")

    args = parser.parse_args()
    
    # multiprocessing.set_start_method('spawn')

    # Creating Processes
    if flag_data['glasses'].value == "STONE":
        process1 = multiprocessing.Process(target=object_detection_process, args=(shared_data, watch_gaze, popup_data, image_buffer, flag_data, shared_array))
        process2 = multiprocessing.Process(target=eye_tracking_process, args=(args.local_ip, args.remote_ip, args.port, args.use_remote, shared_data, popup_data))
        process3 = multiprocessing.Process(target=text_to_speech, args=(watch_gaze, popup_data))
        process5 = multiprocessing.Process(target=scene_image, args=("127.0.0.1", "192.168.1.117", 3425, False, shared_array))
    elif flag_data['glasses'].value == "Nreal":
        process1 = multiprocessing.Process(target=object_detection_process, args=(shared_data, watch_gaze, popup_data, image_buffer, flag_data, shared_array))
        process2 = multiprocessing.Process(target=eye_tracking_process, args=(args.local_ip, args.remote_ip, args.port, args.use_remote, shared_data, popup_data))
        process3 = multiprocessing.Process(target=text_to_speech, args=(watch_gaze, popup_data))
        process4 = multiprocessing.Process(target=popup_windiow, args=(popup_data, image_buffer, flag_data))
    

    # Start Processes
    if flag_data['glasses'].value == "STONE":
        process1.start()
        process2.start()
        process3.start()
        process5.start()
    elif flag_data['glasses'].value == "Nreal":
        process1.start()
        process2.start()
        process3.start()
        process4.start()

    # Join Processes
    if flag_data['glasses'].value == "STONE":
        process1.join()
        process2.join()
        process3.join()
        process5.join()
    elif flag_data['glasses'].value == "Nreal":
        process1.join()
        process2.join()
        process3.join()
        process4.join()
    


if __name__ == "__main__":
    main()
