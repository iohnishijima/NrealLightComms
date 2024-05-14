import numpy as np
import pygame

class TextToSpeech:
    def __init__(self, watch_gaze, popup_data):
        self.watch_gaze = watch_gaze
        self.popup_data = popup_data
        self.check_gaze = {"object_name": None, "is_watching": False, "timestamp": None}
        pygame.mixer.init()
        
    def update_gaze_info(self):
        if self.watch_gaze["is_watching"].value:
            current_object = self.watch_gaze["object_name"].value
            current_timestamp = self.watch_gaze["timestamp"].value

            if self.check_gaze["object_name"] != current_object or self.check_gaze["object_name"] is None:
                self.check_gaze["object_name"] = current_object
                self.check_gaze["is_watching"] = True
                self.check_gaze["timestamp"] = current_timestamp
            elif (current_timestamp - self.check_gaze["timestamp"] >= 0.75):
                self.check_gaze = {"object_name": None, "is_watching": False, "timestamp": None}
                self.popup_data["Confirmation"].value =True
                self.play_sound(current_object)
        else:
            self.check_gaze = {"object_name": None, "is_watching": False, "timestamp": None}

    def play_sound(self, object_name):
        audio_file = f"./audios/{object_name}.mp3"
        if audio_file:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        else:
            print("Audio file not found for the object:", object_name)

    def speech(self):
        try:
            while True:
                if self.popup_data["Confirmation"].value == True and self.popup_data["AddAllData"].value == True:
                    continue
                self.update_gaze_info()
        except KeyboardInterrupt:
            print("Stopping Text to Speech")