import os
import cv2
import copy
import yaml
import argparse
import numpy as np

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(vimg, (x, y), 3, (255, 0, 0), -1)
        points.append((x, y))
        if len(points) > 1:
            cv2.line(vimg, points[-2], points[-1], (255, 0, 0), 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='./images/draw_rec_test.jpg')
    parser.add_argument('-r', '--rtsp_url', type=str, default='')
    args = parser.parse_args()
    
    if args.file:
        img = cv2.imread(args.file)
    
    vimg = copy.deepcopy(img)
    
    # start drawing
    cv2.namedWindow('image')
    bbox = [None, None, None, None]
    cv2.setMouseCallback('image', mouse_callback, [bbox, img])
    
    while(1):
        cv2.imshow('image',vimg)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('b'):
            if len(points) > 0:
                vimg = copy.deepcopy(img)
                points.pop()
                if len(points) > 0:
                    cv2.circle(vimg, points[0], 3, (255, 0, 0), -1)
                for i in range(1, len(points)):
                    cv2.circle(vimg, points[i], 3, (255, 0, 0), -1)
                    cv2.line(vimg, points[i-1], points[i], (255, 0, 0), 1)
        elif key == 13: # enter
            h, w, _ = vimg.shape
            for i, pt in enumerate(points):
                points[i] = [round(pt[0] / w, 4), round(pt[1] / h, 4)]
            points = np.array(points).flatten().tolist()
            print(points)
            break
        elif key == 27:
            break
    cv2.destroyAllWindows()
