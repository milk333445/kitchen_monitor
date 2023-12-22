import cv2
import threading
import queue
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
import time
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import csv
from config import load_config
from utils import normalize_keypoints, draw_polygon_on_image, return_cooking_roi, calculate_iou, cv2ImgAddText, process_detection_frame
import pickle
import copy
import threading
import queue
import warnings
warnings.filterwarnings("ignore")



config = load_config()
bbox_color = tuple(config['bbox_color'])
bbox_thickness = config['bbox_thickness']
bbox_labelstr = config['bbox_labelstr']
kpt_color_map = config['kpt_color_map']
kpt_labelstr = config['kpt_labelstr']
skeleton_map = config['skeleton_map']
normalized_cooking_roi = config['cooking_roi']
loading_bar_color = tuple(config['loading_bar_color'])
loading_bar_radious = tuple(config['loading_bar_radious'])
loading_bar_labelstr = config['loading_bar_labelstr']




class Capture:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.q = queue.Queue()
        self.running = True

    def start_capture(self):
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def get_frame(self):
        if self.q.empty():
            return None
        return self.q.get()

    def release(self):
        self.running = False
        self.cap.release()


def detect_stream(
    source=0, # 不要放影片，會跑很快，且會有問題
    pretrained_model_path='./yolov8s-pose.pt', 
    detect_model_path='./body_language.pkl',
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    data=None,
    imgsz=[640, 640],# yolov8x-pose-p6有時候要改1280，1280，如果有不match的話就改
    conf_thres=0.25,
    iou_thres=0.7,
    bbox=True,
    kpt=True,
    skeleton=True,
    fps16=False,
    show_id=False,
    show_fps=True,
    show_detection=True,
    detect_threshold=0.5,
    confidence_threshold=10 
):
    
    
    # model
    model = AutoBackend(
        weights=pretrained_model_path,
        device=device,
        fp16=False,
        verbose=False
    )
    _ = model.eval()
    
    # detect model
    with open(detect_model_path, 'rb') as f:
        detect_model = pickle.load(f)
    
    camera = Capture(source)
    camera.start_capture()
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                try:
                    cooking_roi, roi_tlrb = return_cooking_roi(frame, normalized_cooking_roi)
                    frame = draw_polygon_on_image(frame, cooking_roi, color=(0, 0, 255), alpha=0.5)
                    frame = process_detection_frame(
                            frame,
                            model,
                            detect_model,
                            device,
                            imgsz=imgsz,
                            conf_thres=conf_thres,
                            iou_thres=iou_thres,
                            data=data,
                            bbox=bbox,
                            kpt=kpt,
                            skeleton=skeleton,
                            fps16=fps16,
                            show_id=show_id,
                            show_fps=show_fps,
                            show_detection=show_detection,
                            detect_threshold=detect_threshold, 
                            confidence_threshold=confidence_threshold,
                            roi_coordinates = roi_tlrb
                        )
                except Exception as e:
                    print('error', e)
                    pass
                resized_frame = cv2.resize(frame, (640, 480))
                cv2.imshow("RTSP Stream", resized_frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()





def main():
    parser = argparse.ArgumentParser(description='Stream detection')
    parser.add_argument('--source', type=int, default=0, help='Video source (0 for default camera or rtsp link)')
    parser.add_argument('--pretrained_model_path', type=str, default='./yolov8s-pose.pt', help='Path to pretrained model')
    parser.add_argument('--detect_model_path', type=str, default='./body_language.pkl', help='Path to detect model')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[640, 640], help='Image size [width, height]')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='Confidence threshold for detection')
    parser.add_argument('--iou_thres', type=float, default=0.7, help='IOU threshold for detection')
    parser.add_argument('--fps16', action='store_true', help='Use FP16 inference')
    parser.add_argument('--show_fps', action='store_true', help='Show FPS on frame')
    parser.add_argument('--detect_threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--confidence_threshold', type=int, default=10, help='Confidence threshold')

    args = parser.parse_args()

    detect_stream(
        source=args.source,
        pretrained_model_path=args.pretrained_model_path,
        detect_model_path=args.detect_model_path,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        detect_threshold=args.detect_threshold,
        confidence_threshold=args.confidence_threshold
    )


if __name__ == '__main__':
    main()
