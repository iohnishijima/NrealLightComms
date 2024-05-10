import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO, FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from ultralytics.models.sam import Predictor as SAMPredictor

class ObjectDetectionProcessor:
    def __init__(self, shared_data, watch_gaze, popup_data, image_buffer):
        self.shared_data = shared_data
        self.watch_gaze = watch_gaze
        self.popup_data = popup_data
        self.display2D = True
        self.objectDetectionFlag = False
        self.segmentAnythingFlag = True
        self.segmentationFlag = False
        self.showFrameFlag = True
        self.modelURL = "/home/ioh_nishijima/デスクトップ/IohNishijima_research/ResearchUEF/NrealLightComms/models/yolov8"
        self.modelSize = ["x", "l", "m", "s", "n",]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.offsetX2d, self.offsetY2d = 40, 110
        self.leftFrameX3d, self.leftFrameY3d, self.RightFrameX3d, self.RightFrameY3d = -5, 150, 18, 150 
        self.image_buffer = image_buffer

        # Load the model
        modelName = f"{self.modelURL}{self.modelSize[0]}{'-seg' if self.segmentationFlag else ''}.pt"
        self.model = YOLO(modelName).to(self.device)
        
        ###change
        modelNameSAM = '/home/ioh_nishijima/デスクトップ/IohNishijima_research/ResearchUEF/NrealLightComms/bus.jpg'
        # Create a FastSAM model
        modelNameSAM = 'ResearchUEF/NrealLightComms/models/FastSAM-s.pt'
        self.samModel = FastSAM(modelNameSAM).to(self.device)
        
        
        self.overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        self.predictor = SAMPredictor(overrides=self.overrides)
        ### here

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.showFrameFlag:
            cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow('Camera', 1920, 100)
    def clip_and_zoom_image(self, image, offset_x, offset_y):
        zoom_factor = 1.05
        target_width = int(1920 / zoom_factor)
        target_height = int(1080 / zoom_factor)
        start_x = max(0, min(image.shape[1] - target_width, (image.shape[1] - target_width) // 2 + offset_x))
        start_y = max(0, min(image.shape[0] - target_height, (image.shape[0] - target_height) // 2 + offset_y))
        clipped_image = image[start_y:start_y + target_height, start_x:start_x + target_width]
        zoomed_image = cv2.resize(clipped_image, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        return zoomed_image

    def process_frame(self, frame):
        GazeX = int(self.shared_data['GazeX'].value * 1920)
        GazeY = int(self.shared_data['GazeY'].value * 1080)
        enter_flag = False
        
        if self.segmentAnythingFlag:
            if 0 <= GazeX < 1920 and 0 <= GazeY < 1080:
                # self.predictor.set_image(frame)  # set with np.ndarray
                # results = self.predictor(points=[GazeX, GazeY], labels=[1])
                # mask = (results[0].masks.data[0].cpu().numpy().astype(np.uint8)) * 255
                # frame = cv2.bitwise_and(frame, frame, mask=mask)
                # self.predictor.reset_image()
                
                everything_results = self.samModel(frame, device='cuda', retina_masks=True, conf=0.4, verbose=False)
                prompt_process = FastSAMPrompt(frame, everything_results, device='cuda')
                ann = prompt_process.point_prompt(points=[[GazeX, GazeY]], pointlabel=[1])
                # print(ann)
                if ann[0].masks is not None:
                    mask = ann[0].masks.data[0].to(torch.uint8).numpy() * 255
                    colored_part = cv2.bitwise_and(frame, frame, mask=mask)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # 3チャネルに戻す
                    inverse_mask = cv2.bitwise_not(mask)

                    # グレイスケール画像に逆マスクを適用
                    grayscale_part = cv2.bitwise_and(gray_frame, gray_frame, mask=inverse_mask)

                    # カラー部分とグレイスケール部分を組み合わせ
                    final_frame = cv2.add(colored_part, grayscale_part)
                    # frame = cv2.bitwise_and(frame, frame, mask=mask)
                else:
                    final_frame = frame
            else:
                final_frame = frame
        elif self.objectDetectionFlag:
            results = self.model(frame, show=False, device=self.device, imgsz=320, verbose=False, half=True, stream=True)
            for r in results:
                classes = r.boxes.cls.cpu().numpy()
                bboxes = r.boxes.xyxy.cpu().numpy()
                for i in range(len(classes)):
                    if bboxes[i][0] <= GazeX <= bboxes[i][2] and bboxes[i][1] <= GazeY <= bboxes[i][3]:
                        label = r.names[classes[i]]
                        bbox = bboxes[i]
                        enter_flag = True
                        self.watch_gaze["object_name"].value = label
                        self.watch_gaze["is_watching"].value = True
                        self.watch_gaze["timestamp"].value = time.time()
                        if self.popup_data["Confirmation"].value == True:
                            np.copyto(self.image_buffer, frame)
                            self.popup_data["X"].value, self.popup_data["Y"].value, self.popup_data["W"].value, self.popup_data["H"].value  = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
                            self.popup_data["AddAllData"].value = True
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if not enter_flag:
                self.watch_gaze["is_watching"].value = False
            final_frame = frame
        else:
            final_frame = frame
        cv2.circle(final_frame, (GazeX, GazeY), 10, (0, 0, 255), -1)
        # return frame
        return final_frame

    def display(self):
        try:
            while True:
                # enter = time.time()
                if self.popup_data["Confirmation"].value == True and self.popup_data["AddAllData"].value == True:
                    continue
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (2592, 1944), interpolation=cv2.INTER_LINEAR)
                clippedFrame = self.clip_and_zoom_image(frame, self.offsetX2d, self.offsetY2d) if self.display2D else np.hstack((self.clip_and_zoom_image(frame, self.leftFrameX3d, self.leftFrameY3d), self.clip_and_zoom_image(frame, self.RightFrameX3d, self.RightFrameY3d)))
                finalFrame = self.process_frame(clippedFrame)
                if self.showFrameFlag:
                    cv2.imshow('Camera', finalFrame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # print(time.time() - enter)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()