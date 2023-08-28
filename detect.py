# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from models.common import DetectMultiBackend
import argparse
import os
import sys
from pathlib import Path
import csv
import torch
import torch.backends.cudnn as cudnn
import dlib
import pandas as pd
import copy
import time
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import threading
# from flask import Flask
import shutil

# Jetson.GPIO
import Jetson.GPIO as GPIO
#Relay = 10
buzzer = 33
led = 7
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
#GPIO.setup(Relay, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(buzzer, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(led, GPIO.OUT, initial=GPIO.LOW)

# Flask Server
def server():
    return
t = threading.Thread(target=server)

# define the name of the directory to be created
### 디렉토리 만드는 코드
if os.path.exists('results/'):
    shutil.rmtree('results/')
os.mkdir('results/')
os.mkdir('results/frames/')
###
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 거리 측정 모델
model_path = None
model = None
poly_features = PolynomialFeatures(degree=2, include_bias=False)

# 거리 예측 함수
def predict(df_test):
    global model

    x_test = np.reshape(df_test['y2'], (-1, 1))
    x_poly = poly_features.fit_transform(x_test)
    distance_pred = model.predict(x_poly)

    df_result = df_test
    df_result['distance'] = -100000

    for idx, row in df_result.iterrows():
        df_result.at[idx, 'distance'] = distance_pred[idx]
    #print(df_result)

    return df_result



@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model='mymodel.pkl',
):
    source = str(source)
### 동영상 저장할 디렉토리 경로 설정하는 코드
###
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    if source == '0':
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1

# object trace var
    currentObjectID = 0
    objectTracker = {}
    preObjectTracker = {}
###

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen = 0
    frame = 0
    cur_time = time.time()
    for path, im, im0s, vid_cap, s in dataset:
        frame += 1
        pre_time = cur_time
        cur_time = time.time()
        print(f'pre_time: {pre_time}, cur_time: {cur_time}')

        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            object_count = 0    # 객체 추적 변수
            seen += 1
            im0 = im0s.copy()

            imc = im0.copy() if save_crop else im0  # for save_crop
            
            # if webcam, list to np.array
            if type(im0) is list:
                for tmp in im0:
                    pass
                im0 = tmp
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            key = 0
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

# trace delete
                objectIDtoDelete = []
                for objectID in objectTracker.keys():
                    trackingQuality = objectTracker[objectID][0].update(im0)
                    if trackingQuality < 10:
                        objectIDtoDelete.append(objectID)

                for objectID in objectIDtoDelete:
                    print('Removing objectID ' + str(objectID) + 'from list of trackers.')
                    objectTracker.pop(objectID, None)
###

                # Write results
                for *xyxy, conf, cls in reversed(det):
# 바운딩 박스치는 코드
#                        annotator.box_label(xyxy, label, color=colors(c, True))
###
                        # xMin yMin xMax yMax
                        csvRowList = []
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())

# for trace code
                        w = x2-x1
                        h = y2-y1

                        x_bar = x1 + 0.5 * w
                        y_bar = y1 + 0.5 * h

                        matchObjectID = None
                        for objectID in objectTracker.keys():
                            trackedPosition = objectTracker[objectID][0].get_position()
                            t_x = int(trackedPosition.left())
                            t_y = int(trackedPosition.top())
                            t_w = int(trackedPosition.width())
                            t_h = int(trackedPosition.height())

                            t_x_bar = t_x + 0.5*t_w
                            t_y_bar = t_y + 0.5*t_h
                            if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x1 <= t_x_bar <= x2) and (y1 <= t_y_bar <= y2)):
                                matchObjectID = objectID

                        if matchObjectID is None:
                            # 추적중인 carID가 아니라면 새 tracker 변수를 만들고
                            # trackedPosition 딕셔너리에 추가
                            print ('Creating new tracker ' + str(currentObjectID))

                            tracker = dlib.correlation_tracker()
                            tracker.start_track(im0, dlib.rectangle(x1, y1, x2, y2))

                            matchObjectID = currentObjectID
                            objectTracker[currentObjectID] = [tracker, -100000]
                            currentObjectID = currentObjectID + 1

###
                        csvRowList = [matchObjectID, y2]
                        
# 2. 거리 예측 (프레임마다 예측할 경우)
                        if object_count == 0:
                            df_test = pd.DataFrame({
                                    'matchObjectID': [matchObjectID],
                                    'y2': [y2]
                                    })
                        else:
                            df_test.loc[object_count] = csvRowList
                        object_count += 1
###
# 해당 프레임 이미지를 저장하는 코드
                        cv2.imwrite('results/frames/{0}.png'.format(frame), im0)
###

# 2. 거리 예측 (프레임마다 예측할 경우)
                df_result = predict(df_test)
                xyxy_tmp = reversed(det).tolist()

                diff_time = cur_time - pre_time
                for idx, row in df_result.iterrows():
                    # 딕셔너리에 객체마다 거리 저장
                    value = objectTracker[int(row[0])]
                    value[1] = row[2]

                    # 거리 출력
                    c = int(cls)
                    annotator.box_label(xyxy_tmp[idx][:4], "dis: "+str(row[2]), color=colors(c, True))
                    cv2.imwrite('results/frames/{0}.png'.format(frame), im0)

                # 속력 계산
                preObjectTrackerKeys = preObjectTracker.keys()
                for key, value in objectTracker.items():
                    if key in preObjectTrackerKeys:
                        preDistance = preObjectTracker[key]
                        curDistance = value[1]

                        speed = (preDistance - curDistance) / diff_time
                        if speed > 0:
                            crashTime = curDistance / speed
                            print(f'\t > crashTime : {crashTime}\n')
                            if crashTime > 0 and crashTime < 30:
                                print(f'\t\tcrashTime: {crashTime}\n')
                                #GPIO.output(Relay, GPIO.LOW)
                                for count in range(3):
                                    GPIO.output(buzzer, GPIO.HIGH)
                                    GPIO.output(led, GPIO.HIGH)
                                    time.sleep(0.5)
                                    GPIO.output(buzzer, GPIO.LOW)
                                    GPIO.output(led, GPIO.LOW)
                                    time.sleep(0.5)
                    else:
                        #GPIO.output(Relay, GPIO.HIGH)
                        pass

                preObjectTracker = {}
                for key, value in objectTracker.items():
                    preObjectTracker[key] = value[1]

                # 현재 프레임 시간을 다음 프레임에서 사용하기 위해 저장

###

            # Stream results
            im0 = annotator.result()
            cv2.imshow('im0', im0)
            key = cv2.waitKey(1)    # 1 millisecond
            if key == ord('q'):
                break
### 동영상으로 저장하는 코드
        if key == ord('q'):
            break



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--model', type=str, default='mymodel.pkl', help='distance model path')
    opt = parser.parse_args()

    global model_path, model
    model_path = parser.parse_args().model
    model = joblib.load(model_path)

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
