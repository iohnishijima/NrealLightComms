import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO, FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from ultralytics.models.sam import Predictor as SAMPredictor
import collections
from translations import translations
from PIL import Image, ImageDraw, ImageFont
import os

class ObjectDetectionProcessor:
    def __init__(self, shared_data, watch_gaze, popup_data, image_buffer, flag_data, image_buffer_scene):
        self.shared_data = shared_data
        self.watch_gaze = watch_gaze
        self.popup_data = popup_data
        self.display2D = True
        self.flag_data = flag_data
        self.circle_color = (0, 0, 255, 0.1)
        self.circle_diameter = 10
        self.image_buffer_scene = image_buffer_scene
        self.translations = translations
        self.translation_flag = True
        self.language_name = "Suomi"
        self.font_path = os.path.join(os.path.dirname(__file__), "Noto_Sans_JP", "NotoSansJP-VariableFont_wght.ttf")
        
        self.modeList = {"OD": "ObjectDetectionByYOLOV8", "SAM": "SegmentAnythingModel", "FastSAM": "FastSegmentAnythingModel", "STONE":"anything"}
        self.mode = self.modeList["STONE"]
        
        self.segmentationFlag = True
        
        self.showFrameFlag = True

        self.gazeX_history = collections.deque(maxlen=5)  # Using 5 points for moving average
        self.gazeY_history = collections.deque(maxlen=5)  # Using 5 points for moving average
        
        self.offsetX2d, self.offsetY2d = 40, 110
        self.image_buffer = image_buffer
        torch.cuda.set_device(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelURL = "/home/ioh_nishijima/デスクトップ/IohNishijima_research/ResearchUEF/NrealLightComms/models/"
        
        if self.mode == self.modeList["OD"]:
            self.modelSize = ["yolov8x", "yolov8l", "yolov8m", "yolov8s", "yolov8n",]
            modelName = f"{self.modelURL}{self.modelSize[0]}{'-seg' if self.segmentationFlag else ''}.pt"
            self.model = YOLO(modelName).to(device=self.device)
            self.process = self.process_frame_OD
        elif self.mode == self.modeList["SAM"]:
            self.overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
            self.model = SAMPredictor(overrides=self.overrides)
            self.process = self.process_frame_SAM
        elif self.mode == self.modeList["FastSAM"]:
            self.modelSize = ["FastSAM-s", "FastSAM-x",]
            modelName = f"{self.modelURL}{self.modelSize[0]}.pt"
            self.model = FastSAM(modelName).to(device=self.device)
            self.process = self.process_frame_FastSAM
        elif self.mode == self.modeList["STONE"]:
            self.modelSize = ["yolov8x", "yolov8l", "yolov8m", "yolov8s", "yolov8n",]
            modelName = f"{self.modelURL}{self.modelSize[0]}{'-seg' if self.segmentationFlag else ''}.pt"
            self.model = YOLO(modelName).to(device=self.device)
            self.process = self.process_frame_STONE

        # Camera setup
        if self.flag_data["glasses"].value == "Nreal":
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.showFrameFlag:
            cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            if self.flag_data["glasses"].value == "Nreal":
                cv2.moveWindow('Camera', 1920, 100)
            elif self.flag_data["glasses"].value == "STONE":
                cv2.moveWindow('Camera', 0, 0)

    # def get_filtered_gaze(self):
    #     rawGazeX = self.shared_data['GazeX'].value * 1920
    #     rawGazeY = self.shared_data['GazeY'].value * 1080
    #     self.gazeX_history.append(rawGazeX)
    #     self.gazeY_history.append(rawGazeY)
    #     filteredGazeX = sum(self.gazeX_history) / len(self.gazeX_history)
    #     filteredGazeY = sum(self.gazeY_history) / len(self.gazeY_history)
    #     return int(filteredGazeX), int(filteredGazeY)   
    def get_filtered_gaze(self):
        rawGazeX = self.shared_data['GazeX'].value * 1920
        rawGazeY = self.shared_data['GazeY'].value * 1080
        
        if rawGazeX != 0 or rawGazeY != 0:  # どちらか一方が0でなければ追加する
            self.gazeX_history.append(rawGazeX)
            self.gazeY_history.append(rawGazeY)
        
        if len(self.gazeX_history) == 0:  # 履歴が空の場合、初期値として0を返す
            return 0, 0
        
        filteredGazeX = sum(self.gazeX_history) / len(self.gazeX_history)
        filteredGazeY = sum(self.gazeY_history) / len(self.gazeY_history)
        return int(filteredGazeX), int(filteredGazeY)     
    
    def clip_and_zoom_image(self, image, offset_x, offset_y):
        zoom_factor = 1.05
        target_width = int(1920 / zoom_factor)
        target_height = int(1080 / zoom_factor)
        start_x = max(0, min(image.shape[1] - target_width, (image.shape[1] - target_width) // 2 + offset_x))
        start_y = max(0, min(image.shape[0] - target_height, (image.shape[0] - target_height) // 2 + offset_y))
        clipped_image = image[start_y:start_y + target_height, start_x:start_x + target_width]
        zoomed_image = cv2.resize(clipped_image, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        return zoomed_image
    

    def put_text_pillow(self, frame, text, position, font_size, color):
        """Pillowを使用してテキストを描画するヘルパー関数"""
        # OpenCVの画像をPillowの画像に変換
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # フォントを定義
        font = ImageFont.truetype(self.font_path, font_size)
        # テキストを描画
        draw.text(position, text, font=font, fill=color)
        # Pillowの画像をOpenCVの画像に戻す
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return frame
    
    def process_frame_OD(self, frame):
        GazeX = int(self.shared_data['GazeX'].value * 1920)
        GazeY = int(self.shared_data['GazeY'].value * 1080)
        enter_flag = False
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
                        self.image_buffer[:] = frame
                        self.popup_data["X"].value, self.popup_data["Y"].value, self.popup_data["W"].value, self.popup_data["H"].value  = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
                        self.popup_data["AddAllData"].value = True
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if not enter_flag:
            self.watch_gaze["is_watching"].value = False
        cv2.circle(frame, (GazeX, GazeY), self.circle_diameter, self.circle_color, -1)
        return frame    
        
    def process_frame_SAM(self, frame):
        GazeX = int(self.shared_data['GazeX'].value * 1920)
        GazeY = int(self.shared_data['GazeY'].value * 1080)
        if 0 <= GazeX < 1920 and 0 <= GazeY < 1080:
            self.model.set_image(frame)
            results = self.model(points=[GazeX, GazeY], labels=[1])
            mask = (results[0].masks.data[0].cpu().numpy().astype(np.uint8)) * 255
            frame = cv2.bitwise_and(frame, frame, mask=mask)
            self.model.reset_image()
        cv2.circle(frame, (GazeX, GazeY), self.circle_diameter, self.circle_color, -1)
        return frame    
        
    def process_frame_FastSAM(self, frame):
        GazeX = int(self.shared_data['GazeX'].value * 1920)
        GazeY = int(self.shared_data['GazeY'].value * 1080)
        if 0 <= GazeX < 1920 and 0 <= GazeY < 1080:
            everything_results = self.model(frame, device='cuda', retina_masks=True, conf=0.4, verbose=False)
            prompt_process = FastSAMPrompt(frame, everything_results, device='cuda')
            ann = prompt_process.point_prompt(points=[[GazeX, GazeY]], pointlabel=[1])
            if ann[0].masks is not None:
                mask = ann[0].masks.data[0].to(torch.uint8).numpy() * 255
                colored_part = cv2.bitwise_and(frame, frame, mask=mask)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                inverse_mask = cv2.bitwise_not(mask)
                grayscale_part = cv2.bitwise_and(gray_frame, gray_frame, mask=inverse_mask)
                frame = cv2.add(colored_part, grayscale_part)
        cv2.circle(frame, (GazeX, GazeY), self.circle_diameter, self.circle_color, -1)
        return frame    
    
    def process_frame_STONE(self, frame):
        # GazeX = int(self.shared_data['GazeX'].value * 1920)
        # GazeY = int(self.shared_data['GazeY'].value * 1080)
        GazeX, GazeY = self.get_filtered_gaze()
        results = self.model(frame, show=False, device=self.device, imgsz=640, verbose=False, half=True, stream=True)
        overlay = frame.copy()
        alpha = 0.4  
        text_overlay = np.zeros_like(frame, dtype=np.uint8)
        for r in results:
            masks = r.masks
            classes = r.boxes.cls.cpu().numpy()
            bboxes = r.boxes.xyxy.cpu().numpy()
            for i in range(len(classes)):
                if bboxes[i][0] <= GazeX <= bboxes[i][2] and bboxes[i][1] <= GazeY <= bboxes[i][3] and r.names[classes[i]] != "person" and  r.names[classes[i]] != "dining table":
                    label = r.names[classes[i]]
                    bbox = bboxes[i]
                    mask_points = masks.xy[i].astype(np.int32)

                    # ラベルの位置を調整して表示
                    label_position = (int(bbox[0]), int(bbox[1]) - 70)
                    translation_text = translations.get(label, {}).get(self.language_name, "N/A")
                    translation_position = (int(bbox[0]), int(bbox[1]) - 100)

                    # マスクとボックスを描画
                    cv2.fillPoly(overlay, [mask_points], color=(0, 255, 0))
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                     # ラベルと翻訳の背景を描画
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 4, 4)
                    font = ImageFont.truetype(self.font_path, 80)
                    translation_size = font.getbbox(translation_text)

                    # ラベル背景
                    cv2.rectangle(overlay, (label_position[0], label_position[1] - label_size[1]), (label_position[0] + label_size[0], label_position[1] + 10), (255, 255, 255, 128), -1)
                    # 翻訳背景
                    cv2.rectangle(overlay, (translation_position[0], translation_position[1] + translation_size[1]), (translation_position[0] + (translation_size[2] - translation_size[0]), translation_position[1] + translation_size[3] + 10), (255, 255, 255, 128), -1)
                    
                    # ラベルと翻訳を描画
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)
                    text_overlay = self.put_text_pillow(text_overlay, translation_text, translation_position, 80, (255, 255, 255))

        # 翻訳機能が有効な場合に左上に表示するテキスト
        if self.translation_flag:
            translation_text = f"Translation Feature Active: {self.language_name}"
            text_size, _ = cv2.getTextSize(translation_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_w, text_h = text_size
            # 背景の位置とサイズ
            bg_rect_start = (5, 5)
            bg_rect_end = (10 + text_w, 10 + text_h)
            cv2.rectangle(frame, bg_rect_start, bg_rect_end, (0, 0, 0, 128), -1)
            # テキストの描画
            cv2.putText(frame, translation_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.circle(overlay, (GazeX, GazeY), self.circle_diameter, self.circle_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.addWeighted(text_overlay, 1, frame, 1, 0, frame)  # 事前に描画されたテキストを重ねる
        return frame

    def display(self):
        try:
            while True:
                # enter1 = time.time()
                # if self.popup_data["Confirmation"].value == True and self.popup_data["AddAllData"].value == True:
                #     continue
                if self.flag_data["glasses"].value == "Nreal":
                    if self.popup_data["Confirmation"].value == True and self.popup_data["AddAllData"].value == True:
                        continue
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    clippedFrame = self.clip_and_zoom_image(frame, self.offsetX2d, self.offsetY2d)
                elif self.flag_data["glasses"].value == "STONE":
                    frame = np.frombuffer(self.image_buffer_scene.get_obj(), dtype=np.uint8).reshape((480, 640, 3))
                    clippedFrame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                # frame = cv2.resize(frame, (2592, 1944), interpolation=cv2.INTER_LINEAR)
                # enterProcess = time.time()
                finalFrame = self.process(clippedFrame)
                # print(f"Process time: {time.time() - enterProcess}")
                if self.showFrameFlag:
                    cv2.imshow('Camera', finalFrame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # print(f"total time: {time.time() - enter1}")
        finally:
            if self.flag_data["glasses"].value == "Nreal":
                self.cap.release()
            cv2.destroyAllWindows()