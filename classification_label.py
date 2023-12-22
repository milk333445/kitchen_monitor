import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
import time
import cv2
from tqdm import tqdm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import csv
from sklearn.model_selection import train_test_split
from utils import normalize_keypoints
from config import load_config
import warnings
warnings.filterwarnings("ignore")

config = load_config()
bbox_color = tuple(config['bbox_color'])
bbox_thickness = config['bbox_thickness']
bbox_labelstr = config['bbox_labelstr']
kpt_color_map = config['kpt_color_map']
kpt_labelstr = config['kpt_labelstr']
skeleton_map = config['skeleton_map']


def process_label_frame(
        img, 
        model,
        device,
        imgsz=[640, 640],
        conf_thres=0.25,
        iou_thres=0.7,
        data=None, 
        bbox=True, 
        kpt=True, 
        skeleton=True, 
        fps16=False,
        show_id=False,
        show_fps=True,
    ):
    """
    針對單張圖像進行處理，包括預處理、預測、後處理
    輸入：
        img_bgr: 原始圖像
        pretrained_model_path: 預訓練模型路徑
        data: 資料集物件yaml(人的話不用)
    """
    
    # 前一時刻的預測結果
    
    start_time = time.time()
    
    # 縮小圖像尺寸
    pre_transform_result = LetterBox(new_shape=imgsz, auto=True)(image=img)
    # 歸一化
    input_tensor = pre_transform_result / 255
    # 增加 batch 維度
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = input_tensor.transpose(0, 3, 1, 2) # 轉換為 [B, C, H, W]
    
    input_tensor = np.ascontiguousarray(input_tensor) # 轉換為 contiguous array，使得記憶體連續，訪問更快
    input_tensor = torch.from_numpy(input_tensor).to(device).float()
    if fps16:
        input_tensor = input_tensor.half()
    
    
    preprocess_time = time.time()
    
    if show_fps:
        print(f'Preprocess time: {preprocess_time - start_time}')
    
    #進行預測
    preds = model(input_tensor)
    
    inference_time = time.time()
    
    if show_fps:
        print(f'Inference time: {inference_time - preprocess_time}')
    
    pred = ops.non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres, nc=1)[0]
    
    # 後處理，處理檢測框
    pred[:, :4] = ops.scale_boxes(pre_transform_result.shape[:2], pred[:, :4], img.shape).round()
    pred_det = pred[:, :6].cpu().numpy() # [x1, y1, x2, y2, conf, cls]
    bboxes_xyxy = pred_det[:, :4].astype('uint32')
    
    pred_kpts = pred[:, 6:].view(len(pred), model.kpt_shape[0], model.kpt_shape[1]).cpu().numpy()
    pred_kpts = ops.scale_coords(pre_transform_result.shape[:2], pred_kpts, img.shape)
    bboxes_keypoints = pred_kpts.astype('uint32')
    
    
    post_process_time = time.time() 
    if show_fps:
        print(f'Post process time: {post_process_time - inference_time}')
    
    num_bbox = len(pred_det)
    
    for idx in range(num_bbox):

        if bbox:
            bbox_xyxy = bboxes_xyxy[idx]
            bbox_label = model.names[0]
            img = cv2.rectangle(
                img,
                (bbox_xyxy[0], bbox_xyxy[1]),
                (bbox_xyxy[2], bbox_xyxy[3]),
                color=bbox_color,
                thickness=bbox_thickness
            )
            
            img = cv2.putText(
                img,
                bbox_label,
                (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                cv2.FONT_HERSHEY_SIMPLEX,
                bbox_labelstr['font_size'],
                bbox_color,
            )
            
            
        
        bbox_keypoints = bboxes_keypoints[idx]
        
        if skeleton:
            # 畫連接線
            for skeleton in skeleton_map:
                srt_kpt_id = skeleton['srt_kpt_id']
                srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
                srt_kpt_y = bbox_keypoints[srt_kpt_id][1]
                
                if srt_kpt_x == 0 and srt_kpt_y == 0:
                    continue
                
                # 獲取終止點
                dst_kpt_id = skeleton['dst_kpt_id']
                dst_kpt_x = bbox_keypoints[dst_kpt_id][0]
                dst_kpt_y = bbox_keypoints[dst_kpt_id][1]
                
                if dst_kpt_x == 0 and dst_kpt_y == 0:
                    continue
                
                skeleton_color = skeleton['color']
                skeleton_thickness = skeleton['thickness']
                
                img = cv2.line(
                    img,
                    (srt_kpt_x, srt_kpt_y),
                    (dst_kpt_x, dst_kpt_y),
                    color=skeleton_color,
                    thickness=skeleton_thickness
                )
        if kpt:
            # 畫關鍵點
            for kpt_id in kpt_color_map:
                
                kpt_color = kpt_color_map[kpt_id]['color']
                kpt_radius = kpt_color_map[kpt_id]['radius']
                kpt_x = bbox_keypoints[kpt_id][0]
                kpt_y = bbox_keypoints[kpt_id][1]
                
                img = cv2.circle(
                    img,
                    (kpt_x, kpt_y),
                    radius=kpt_radius,
                    color=kpt_color,
                    thickness=-1
                )
                
                if show_id:
                    kpt_label = str(kpt_id)
                    img = cv2.putText(
                        img,
                        kpt_label,
                        (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        kpt_labelstr['font_size'],
                        kpt_color,
                        kpt_labelstr['font_thickness']
                    )
    end_time = time.time()
    
    if show_fps:
        FPS = 1 / (end_time - start_time)
        print(f'FPS: {FPS}')
        FPS_string = 'FPS  {:.2f}'.format(FPS)
        img = cv2.putText(
            img,
            FPS_string,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (255, 0, 255),
            2
        )     
    return img, pred_det, pred_kpts
def load_video(
        input_path='./videos/output3.mp4', 
        pretrained_model_path='./yolov8x-pose-p6.pt', 
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
        show_fps=True
    ):
    
    # model
    model = AutoBackend(
    weights=pretrained_model_path,
    device=device,
    fp16=False,
    verbose=False
    )
    _ = model.eval()
    
    
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
    
    
    with tqdm(total=frame_count-1) as pbar:
        frame_count = 0
        write_count = 0
        try:
            while(cap.isOpened()):
                sucess, frame = cap.read()
                frame_count += 1
                if not sucess:
                    break
                
                try:
                    frame, pred_det, pred_kpts = process_label_frame(
                        frame,
                        model,
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
                        show_fps=show_fps
                    )
                    pbar.update(1)
                    try:
                        
                        bounding_box = pred_det[0][:4]
                        normalized_keypoints = normalize_keypoints(pred_kpts[0], bounding_box)
                        normalized_keypoints_list = [item for sublist in normalized_keypoints for item in sublist]
                        if frame_count >= 0 and frame_count <= 330:
                            normalized_keypoints_list.insert(0, 'stand')
                            with open('coords.csv', 'a', newline='') as f:
                                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                csv_writer.writerow(normalized_keypoints_list)
                                write_count += 1
                            
                        elif (frame_count >= 380 and frame_count <= 390) or (frame_count >= 480 and frame_count <= 520) or (frame_count >= 640 and frame_count <= 680):
                            normalized_keypoints_list.insert(0, 'crunch')
                            with open('coords.csv', 'a', newline='') as f:
                                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                csv_writer.writerow(normalized_keypoints_list)
                                write_count += 1
                            
                        elif (frame_count >= 800 and frame_count <= 880) or (frame_count >= 1000 and frame_count <= 1360):
                            normalized_keypoints_list.insert(0, 'temp_measure')
                            with open('coords.csv', 'a', newline='') as f:
                                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                csv_writer.writerow(normalized_keypoints_list)
                                write_count += 1
                        
                        pass
                        
                    except Exception as e:
                        pass
                    
                    frame = cv2.putText(
                        frame,
                        str(frame_count),
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 255),
                    )
                    
                    
                except Exception as e:
                    print('error', e)
                    pass
                if sucess:
                    resized_frame = cv2.resize(frame, (1200, 800))
                    cv2.imshow('frame', resized_frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    
        except:
            print('error')
            pass
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    df = pd.read_csv('coords.csv')
    X = df.drop('class', axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=87)
    
    pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }
    
    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model
        
    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat), f1_score(y_test, yhat, average='weighted'))
        
    with open('body_language.pkl', 'wb') as f:
        pickle.dump(fit_models['rf'], f)