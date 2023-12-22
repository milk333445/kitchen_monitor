import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
import time
from tqdm import tqdm
from config import load_config
from PIL import Image, ImageDraw, ImageFont
import threading
import queue
import pickle
import pandas as pd

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
pre_state = config['pre_state']
high_confidence_count = config['high_confidence_count']
difference_count = config['difference_count']
inside_and_interact_count = config['inside_and_interact_count']


def preprocess_image(image, imgsz, device, fps16=False):
    

    pre_transform_result = LetterBox(new_shape=imgsz, auto=True)(image=image)
    
    # 歸一化
    input_tensor = pre_transform_result / 255
    
    # 增加 batch 維度
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = input_tensor.transpose(0, 3, 1, 2)  # 轉換為 [B, C, H, W]
    
    input_tensor = np.ascontiguousarray(input_tensor)  # 轉換為 contiguous array，使得記憶體連續，訪問更快
    input_tensor = torch.from_numpy(input_tensor).to(device).float()
    
    if fps16:
        input_tensor = input_tensor.half()

    return input_tensor, pre_transform_result

def postprocess_predictions(preds, pre_transform_result, conf_thres, iou_thres, img, model):

    pred = ops.non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres, nc=1)[0]

    # 後處理，處理檢測框
    pred[:, :4] = ops.scale_boxes(pre_transform_result.shape[:2], pred[:, :4], img.shape).round()
    pred_det = pred[:, :6].cpu().numpy() # [x1, y1, x2, y2, conf, cls]
    bboxes_xyxy = pred_det[:, :4].astype('uint32')

    pred_kpts = pred[:, 6:].view(len(pred), model.kpt_shape[0], model.kpt_shape[1]).cpu().numpy()
    pred_kpts = ops.scale_coords(pre_transform_result.shape[:2], pred_kpts, img.shape)
    bboxes_keypoints = pred_kpts.astype('uint32')

    return bboxes_xyxy, bboxes_keypoints, pred_kpts, pred_det


def draw_skeleton_on_image(img, bbox_keypoints):
    for skeleton in skeleton_map:
        srt_kpt_id = skeleton['srt_kpt_id']
        srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
        srt_kpt_y = bbox_keypoints[srt_kpt_id][1]

        if srt_kpt_x == 0 and srt_kpt_y == 0:
            continue

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

    return img

def draw_keypoints_on_image(img, bbox_keypoints, show_id=False):
    for kpt_id in kpt_color_map:
        kpt_color_info = kpt_color_map[kpt_id]
        kpt_color = kpt_color_info['color']
        kpt_radius = kpt_color_info['radius']
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
            offset_x = kpt_color_info.get('offset_x', 0)
            offset_y = kpt_color_info.get('offset_y', 0)
            font_size = kpt_color_info.get('font_size', 0.5)
            font_thickness = kpt_color_info.get('font_thickness', 1)

            img = cv2.putText(
                img,
                kpt_label,
                (kpt_x + offset_x, kpt_y + offset_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                kpt_color,
                font_thickness
            )

    return img


def process_single_frame(
        img, 
        pretrained_model_path, 
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
        show_fps=True
    ):
    """
    針對單張圖像進行處理，包括預處理、預測、後處理
    輸入：
        img_bgr: 原始圖像
        pretrained_model_path: 預訓練模型路徑
        data: 資料集物件yaml(人的話不用)
    """
    model = AutoBackend(
        weights=pretrained_model_path,
        device=device,
        fp16=fps16,
        verbose=False,
        data=data
    )
    
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
                    #kpt_label = kpt_color_map[kpt_id]['name']
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
    return img   


def process_frame(
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
        show_fps=True
    ):
    """
    針對單張圖像進行處理，包括預處理、預測、後處理
    輸入：
        img_bgr: 原始圖像
        pretrained_model_path: 預訓練模型路徑
        data: 資料集物件yaml(人的話不用)
    """
    
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
                    #kpt_label = kpt_color_map[kpt_id]['name']
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
    return img  



def generate_video(
        input_path='./videos/test1.mp4', 
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
    filehead = input_path.split('/')[-1]
    output_path = "8xp6out-" + filehead
    
    
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
                    frame = process_frame(
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
                except Exception as e:
                    print('error', e)
                    pass
                if sucess == True:
                    out.write(frame)
                    pbar.update(1)
        except:
            print('error')
            pass
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    print('save video done', output_path)
    

def capture_rtsp_usb(
    rtsp_link,
    pretrained_model_path='./yolov8n-pose.pt',
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
    
    cap = cv2.VideoCapture(rtsp_link)
    
    while(cap.isOpened()):
        sucess, frame = cap.read()
        if not sucess:
            print('cannot get frame from rtsp')
            break
        try:
            frame = process_frame(
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
        except Exception as e:
            print('error', e)
            pass
        if sucess == True:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    cap.release()
    


def normalize_keypoints(keypoints, bounding_box):
    x1, y1, x2, y2 = bounding_box
    box_width = x2 - x1
    box_height = y2 - y1
    
    normalized_keypoints = []
    for kpt in keypoints:
        x, y, conf = kpt
        normalized_x = (x - x1) / box_width
        normalized_y = (y - y1) / box_height
        normalized_keypoints.append([normalized_x, normalized_y, conf])
    return normalized_keypoints

def return_cooking_roi(image, cooking_roi):
    image_height, image_width, _ = image.shape
    coordinates = []
    for i in range(0, len(cooking_roi), 2):
        x = int(cooking_roi[i] * image_width)
        y = int(cooking_roi[i + 1] * image_height)
        coordinates.append([x, y])
    
    top_left_right_bottom = [coordinates[0][0], coordinates[0][1], coordinates[2][0], coordinates[2][1]]
    
    return coordinates, top_left_right_bottom

def draw_polygon_on_image(image, coords, color, alpha):
    overlay = image.copy()
    pts = np.array(coords, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


def calculate_iou(bbox, roi):
    
    x1_intersect = max(bbox[0], roi[0])
    y1_intersect = max(bbox[1], roi[1])
    x2_intersect = min(bbox[2], roi[2])
    y2_intersect = min(bbox[3], roi[3])
    
    # intersection area
    width_intersect = max(0, x2_intersect - x1_intersect + 1)
    height_intersect = max(0, y2_intersect - y1_intersect + 1)
    area_intersect = width_intersect * height_intersect
    
    # bbox area
    area_bbox = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    # roi area
    area_roi = (roi[2] - roi[0] + 1) * (roi[3] - roi[1] + 1)
    # 計算交集占roi面積的比例
    return area_intersect / float(area_roi)

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def process_detection_frame(
        img, 
        model,
        detect_model,
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
        show_detection=True,
        detect_threshold=0.8,
        confidence_threshold=10,
        roi_coordinates=None
    ):
    """
    針對單張圖像進行處理，包括預處理、預測、後處理
    輸入：
        img_bgr: 原始圖像
        pretrained_model_path: 預訓練模型路徑
        detect_model: 偵測模型
        data: 資料集物件yaml(人的話不用)
    """
    
    # 前一時刻的預測結果
    global pre_state, high_confidence_count, difference_count, inside_and_interact_count
    
    
    start_time = time.time()
    input_tensor, pre_transform_result = preprocess_image(img, imgsz, device, fps16)
    preprocess_time = time.time()
    
    if show_fps:
        print(f'Preprocess time: {preprocess_time - start_time}')
    
    #進行預測
    
    preds = model(input_tensor)
    
    inference_time = time.time()
    
    if show_fps:
        print(f'Inference time: {inference_time - preprocess_time}')
   
    
    bboxes_xyxy, bboxes_keypoints, pred_kpts, pred_det = postprocess_predictions(
        preds, 
        pre_transform_result,
        conf_thres,
        iou_thres,
        img,
        model
    )
    
    
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
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            
            
            if show_detection:
                keypoint = pred_kpts[idx]
                
                center = (int(pred_kpts[idx][0][0]), int(pred_kpts[idx][0][1]))# nose
                
                normalized_keypoints = normalize_keypoints(keypoint, bbox_xyxy)
                normalized_keypoints_list = [item for sublist in normalized_keypoints for item in sublist]
                
                # 預測
                X = pd.DataFrame([normalized_keypoints_list])
                body_language_class = detect_model.predict(X)[0]
                body_language_prob = detect_model.predict_proba(X)[0]
                
               
                if body_language_prob[np.argmax(body_language_prob)] > detect_threshold:
                    high_confidence_count += 1
                    
                    if body_language_class != pre_state and pre_state != 'preparing':
                        difference_count += 1
                    
                    if high_confidence_count > confidence_threshold:
                        if difference_count < 5 and pre_state != 'preparing':
                            body_language_class = pre_state
                        else:
                            pre_state = body_language_class
                            difference_count = 0
                    else:   
                        body_language_class = pre_state
                    
                    
                        
                    img = cv2.rectangle(
                        img,
                        pt1=(int(bbox_xyxy[0]), int(bbox_xyxy[1] + 10)),
                        pt2=(int(bbox_xyxy[0] + len(body_language_class) * 20), int(bbox_xyxy[1] +30)),
                        color=(255, 0, 0),
                        thickness=-1
                    )
                    
                    img = cv2.putText(
                        img,
                        body_language_class,
                        (int(bbox_xyxy[0]), int(bbox_xyxy[1] + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA  
                    )
                    
                    
                    
                    
                    img = cv2.putText(
                        img,
                        str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                        (int(bbox_xyxy[0] + len(body_language_class) * 20 + 10), int(bbox_xyxy[1] + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )    
                    
                    # 算interact/roi
                    if roi_coordinates is not None:
                        area = calculate_iou(bbox_xyxy, roi_coordinates)
                        
                        area_scale = min(area/0.8, 1)
                        
                        img = cv2.putText(
                            img,
                            str(round(area_scale, 2)),
                            (int(roi_coordinates[0]), int(roi_coordinates[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )
                        
                        # 如果在ROI內且是量測溫度的動作，做額外處理
                        if area_scale > 0.8 and body_language_class == 'temp_measure':
                            inside_and_interact_count += 1
                            
                            if inside_and_interact_count > 70:
                                inside_and_interact_count = 70
                        
                            process = min(inside_and_interact_count, loading_bar_labelstr['max_count_frame'])
                            angle = 360 * process / loading_bar_labelstr['max_count_frame']
                            
                            if angle > 0:
                                img = cv2.ellipse(
                                        img,
                                        center,
                                        loading_bar_radious,
                                        0, 
                                        0, 
                                        angle,
                                        loading_bar_color,
                                        loading_bar_labelstr['font_thickness']   
                                    ) 
                                
                            if angle == 360:
                                # 寫文字start temp measure
                                
                                img = cv2.ellipse(
                                        img,
                                        center,
                                        loading_bar_radious,
                                        0, 
                                        0, 
                                        angle,
                                        (255, 165, 0),
                                        loading_bar_labelstr['font_thickness']   
                                    ) 
                                
                                
                                img = cv2.rectangle(
                                    img,
                                    pt1=(int(bbox_xyxy[0]), int(bbox_xyxy[1] - 40)),
                                    pt2=(int(bbox_xyxy[0] + len('start temp measure') * 20), int(bbox_xyxy[1] - 5)),
                                    color=(0, 0, 255),
                                    thickness=-1
                                    
                                )
                                
                                img = cv2.putText(
                                    img,
                                    'start temp measure',
                                    (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA
                                )
                                
                                
                        else:
                            inside_and_interact_count = max(inside_and_interact_count - 1, 0)
                            process = min(inside_and_interact_count, loading_bar_labelstr['max_count_frame'])
                            angle = 360 * process / loading_bar_labelstr['max_count_frame']
                            if angle > 0:
                                img = cv2.ellipse(
                                        img,
                                        center,
                                        loading_bar_radious,
                                        0, 
                                        0,
                                        angle,
                                        loading_bar_color,
                                        loading_bar_labelstr['font_thickness']   
                                    )
                                              
                elif body_language_prob[np.argmax(body_language_prob)] > 0.6 and body_language_prob[np.argmax(body_language_prob)] < detect_threshold and high_confidence_count > 30:
                    high_confidence_count = 0
                    img = cv2.rectangle(
                        img,
                        pt1=(int(bbox_xyxy[0]), int(bbox_xyxy[1] + 10)),
                        pt2=(int(bbox_xyxy[0] + len(pre_state) * 20), int(bbox_xyxy[1] +30)),
                        color=(255, 0, 0),
                        thickness=-1
                    )
                    img = cv2.putText(
                        img,
                        pre_state,
                        (int(bbox_xyxy[0]), int(bbox_xyxy[1] + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA  
                    )
                    img = cv2.putText(
                        img,
                        str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                        (int(bbox_xyxy[0] + len(pre_state) * 20 + 10), int(bbox_xyxy[1] + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA
                    )    
                else:
                    inside_and_interact_count = 0
                    
            
        
        bbox_keypoints = bboxes_keypoints[idx]
        
        if skeleton:
            # 畫連接線
            
            img = draw_skeleton_on_image(
                img,
                bbox_keypoints
            )
            
           
        if kpt:
            # 畫關鍵點
            img = draw_keypoints_on_image(
                img,
                bbox_keypoints,
                show_id
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
    return img

if __name__ == '__main__':
    generate_video()