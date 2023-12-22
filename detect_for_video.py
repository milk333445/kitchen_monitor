import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
import time
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
import argparse
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





def detect_video(
        input_path='./videos/output3.mp4', 
        pretrained_model_path='./yolov8m-pose.pt', 
        detect_model_path='./body_language.pkl',
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        data=None,
        imgsz=[640, 640], 
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
    
    filehead = input_path.split('/')[-1]
    output_path = "8mout-" + filehead
    
    
    
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
    
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while(cap.isOpened()):
        sucess, frame = cap.read()
        frame_count += 1
        if not sucess:
            break
    cap.release()
    print(f'Frame count: {frame_count}')
    
    cap = cv2.VideoCapture(input_path)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (int(frame_size[0]), int(frame_size[1]))
    )
    
    
    with tqdm(total=frame_count-1) as pbar:
        try:
            while(cap.isOpened()):
                sucess, frame = cap.read()
      
                if not sucess:
                    break
                
                try:
                    # 畫 ROI
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
                    pbar.update(1)
                    
                except Exception as e:
                    print('error', e)
                    pass
                if sucess:
                    resized_frame = cv2.resize(frame, (1200, 800))
                    cv2.imshow('frame', resized_frame)
                    out.write(frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                
                    
        except:
            print('error')
            pass
    cv2.destroyAllWindows()
    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description='Video detection')
    parser.add_argument('--input_path', type=str, default='./videos/output3.mp4', help='Path to input video file')
    parser.add_argument('--pretrained_model_path', type=str, default='./yolov8m-pose.pt', help='Path to pretrained model')
    parser.add_argument('--detect_model_path', type=str, default='./body_language.pkl', help='Path to detect model')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[640, 640], help='Image size [width, height]')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='Confidence threshold for detection')
    parser.add_argument('--iou_thres', type=float, default=0.7, help='IOU threshold for detection')
    parser.add_argument('--fps16', action='store_true', help='Use FP16 inference')
    parser.add_argument('--show_fps', action='store_true', help='Show FPS on frame')
    parser.add_argument('--detect_threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--confidence_threshold', type=int, default=10, help='Confidence threshold')

    args = parser.parse_args()

    detect_video(
        input_path=args.input_path,
        pretrained_model_path=args.pretrained_model_path,
        detect_model_path=args.detect_model_path,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        detect_threshold=args.detect_threshold,
        confidence_threshold=args.confidence_threshold,
        
    )



if __name__ == '__main__':
    main()
    