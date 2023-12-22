# kitchen monitor demo
- In this research, we use the YOLOv8 pose detection algorithm to identify the chef's body frame and 17 key points for precise action recognition. These actions are primarily categorized into three classes: standing, crouching, and temperature measurement. Subsequently, these actions are classified and recognized using a Random Forest classifier.
- We have achieved real-time monitoring of the chef's status and accurate identification of the chef when they move to the cooking location and begin measuring food temperature. Once this action is recognized, the camera interacts with the chef, providing relevant information. 


https://github.com/milk333445/kitchen_monitor/assets/72238891/78cf8a7e-ed27-4fd4-86b9-4831f3fdd63c



## files explanation
- main files
  - classification_label.py: 
    - This file primarily serves to collect and train datasets for classification tasks. It operates on video units; for example, if you want to label data for the "standing" action, it requires videos where the subject is exclusively standing. The functions within this script automatically capture the coordinates of 17 key points (normalized according to frame width and height) and write them to a CSV file in row format.
  - detect_multithread.py: 
    - The main purpose of this file is to achieve real-time streaming pose detection. It displays the current state of individuals in the stream, whether they have entered the temperature measurement area for temperature assessment, and provides real-time feedback on the screen.
  - detect_for_video.py: 
    - Similar to the above description, this file is designed for video files. It serves the same purpose as "detect_multithread.py." Since multithreading is not suitable for video files, it is separated to ensure smooth functionality. Its features align with those of "detect_multithread.py."

## paramer explanation
Here are the parameters you can configure in the detect_multithread.py script:

- source: Specify the video source. You can use either the default camera (0) or an RTSP link to your video source.
- pretrained_model_path: Set the path to the pretrained YOLOv8 pose model.
- detect_model_path: Provide the path to the detection model, which is used for action classification.
- imgsz: Define the image size for processing in the format [width, height].
- conf_thres: Adjust the confidence threshold for detection. This threshold determines whether an object has been detected.
- iou_thres: Set the IOU (Intersection over Union) threshold for detection. It controls the overlap between bounding boxes.
- fps16: Enable this option to use FP16 inference, which can accelerate processing but may affect precision.
- detect_threshold: Specify the detection threshold for action recognition.
- confidence_threshold: Set the confidence threshold for action classification(inorder to let the output stable).
## quick start
```python=
git clone https://github.com/milk333445/kitchen_monitor.git
cd kitchen_monitor
```
- Prepare the official pre-trained weights for YOLOv8 pose.
- Prepare the classifier weights (which can be trained using the classification_label.py script).
- Ensure that the necessary packages are installed.
- Confirm that the parameters in detect_multithread.py are correctly filled out.
- Run the following command:
```python=
python3 detect_multithread.py \
    --source  \
    --pretrained_model_path  \
    --detect_model_path  \
    --imgsz  \
    --conf_thres  \
    --iou_thres  \
    --fps16 \
    --detect_threshold  \
    --confidence_threshold 

```

## time comaprison
- setting:
  - people count:1
  - batch_size:1
  - input_size:640, 640



|                       | YOLOv8n | YOLOv8s | YOLOv8m | YOLOv8l | YOLOv8x | YOLOv8x-pose-p6 |
|:--------------------- | ------- |:------- |:------- |:------- |:------- |:--------------- |
| Preprocess time(sec)  | 0.003   | 0.0033  | 0.0031  | 0.0032  | 0.0029  | 0.00289         |
| Inference time(sec)   | 0.0073  | 0.00828 | 0.0098  | 0.011   | 0.0114  | 0.0149          |
| Postprocess time(sec) | 0.0011  | 0.0011  | 0.0044  | 0.0126  | 0.0206  | 0.0206          |
| detect time(sec)      | 0.0146  | 0.0178  | 0.015   | 0.0147  | 0.017   | 0.0158          |
| FPS(sec)              | 37.68   | 29.95   | 30.29   | 23.75   | 19.05   | 18.255          |

