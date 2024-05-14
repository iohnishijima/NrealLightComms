import cv2
import numpy as np
from screeninfo import get_monitors
import time

class Confirmation:
    def __init__(self, popup_data, image_buffer, flag_data):
        self.flag_data = flag_data
        if self.flag_data["glasses"].value == "Nreal":
            target_monitor = get_monitors()[1]
        elif self.flag_data["glasses"].value == "STONE":
            target_monitor = get_monitors()[0]
        self.window_width = 1024
        self.window_height = 768
        self.x = target_monitor.x + (target_monitor.width - self.window_width) // 2
        self.y = target_monitor.y + (target_monitor.height - self.window_height) // 2
        self.popup_data = popup_data
        self.image_buffer = image_buffer
        self.button_pressed = None

    def show_popup(self):
        # Create a black image
        img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        # Display the message
        cv2.putText(img, "Do you want to go to this object?", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Process and display the image from buffer
        image_array = np.copy(self.image_buffer)
        object = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        object = object[int(self.popup_data["Y"].value):int(self.popup_data["H"].value), int(self.popup_data["X"].value):int(self.popup_data["W"].value)]
        resized_object = cv2.resize(object, (640, 480), interpolation=cv2.INTER_LINEAR)
        img[150:630, 192:832] = resized_object

        # Draw buttons
        cv2.rectangle(img, (50, 650), (350, 750), (0, 255, 0), -1)
        cv2.putText(img, "Confirm", (70, 720),  cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (700, 650), (1000, 750), (0, 0, 255), -1)
        cv2.putText(img, "Cancel", (720, 720),  cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

        # Setup the window
        cv2.namedWindow("Confirmation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Confirmation", self.window_width, self.window_height)
        cv2.moveWindow("Confirmation", self.x, self.y - 200)
        
        select_button = {"button": None, "time": None, "break": False}

        # Show image
        while True:
            img_copy = img.copy()
            gaze_x = int(self.popup_data['GazeX'].value * 1920)
            gaze_y = int(self.popup_data['GazeY'].value * 1080)
            if (self.x - 1920) <= gaze_x <= (self.x + self.window_width - 1920) and self.y <= gaze_y <= self.y + self.window_height:
                cv2.circle(img_copy, (gaze_x-(self.x - 1920), gaze_y), 20, (255, 0, 0), -1)
            cv2.imshow("Confirmation", img_copy)
            cv2.waitKey(1)
            if 50 <= gaze_x-(self.x - 1920) <= 350 and 650 <= gaze_y <= 750:  # Confirm button area
                if select_button["time"] == None or select_button["button"] != "Confirm":
                    select_button["time"] = time.time()
                select_button["button"] = "Confirm"
                if select_button["button"] == "Confirm" and (time.time() - select_button["time"]) >= 1:
                    self.button_pressed = 'Confirm'
                    cv2.destroyAllWindows()
                    select_button["break"] = True
            elif 700 <= gaze_x-(self.x - 1920) <= 1000 and 650 <= gaze_y <= 750:  # Cancel button area
                if select_button["time"] == None or select_button["button"] != "Cancel":
                    select_button["time"] = time.time()
                select_button["button"] = "Cancel"
                if select_button["button"] == "Cancel" and (time.time() - select_button["time"]) >= 1:
                    self.button_pressed = 'Cancel'
                    cv2.destroyAllWindows()
                    select_button["break"] = True
            if select_button["break"] == True:
                break

    def main_loop(self):
        while True:
            if self.popup_data["Confirmation"].value == True and self.popup_data["AddAllData"].value == True:
                self.show_popup()
                print(self.button_pressed)
                self.popup_data["Confirmation"].value = False
                self.popup_data["AddAllData"].value = False
