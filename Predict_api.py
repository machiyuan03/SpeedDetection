#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from flask import Flask, render_template, Response, request
import json
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.checks import cv2, print_args
from utils.general import update_options

import supervision as sv

# Initialize paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Initialize Flask API
app = Flask(__name__, static_folder='static')

# Define constants for perspective transform
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]])


# Define ViewTransformer class
# Define ViewTransformer class
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def predict(opt):
    """
    Perform object detection using the YOLO model and yield results.

    Parameters:
    - opt (Namespace): A namespace object that contains all the options for YOLO object detection,
        including source, model path, confidence thresholds, etc.

    Yields:
    - JSON: If opt.save_txt is True, yields a JSON string containing the detection results.
    - bytes: If opt.save_txt is False, yields JPEG-encoded image bytes with object detection results plotted.
    """
    model = YOLO(opt.model)
    results = model(**vars(opt), stream=True)

    tracker = sv.ByteTrack(frame_rate=30)  # Assuming 30 FPS for simplicity
    view_transformer = ViewTransformer(SOURCE, TARGET)
    coordinates = defaultdict(lambda: deque(maxlen=30))

    for result in results:
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        labels = []
        colors = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < 15:  # Half a second at 30 FPS
                labels.append(f"#{tracker_id}")
                colors.append((0, 255, 0))  # Green for low speed
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance_pixels = abs(coordinate_start - coordinate_end)

                # Convert pixel distance to meters
                pixel_per_meter = TARGET_HEIGHT / TARGET_WIDTH  # Assuming TARGET_HEIGHT and TARGET_WIDTH are in meters
                distance_meters = distance_pixels / pixel_per_meter

                time_seconds = len(coordinates[tracker_id]) / 30.0
                speed_kmph = distance_meters / time_seconds * 0.38
                labels.append(f"#{tracker_id} {int(speed_kmph)} km/h")

                # Choose color based on speed
                if speed_kmph > 60:
                    colors.append((0, 0, 0))  # black for high speed
                else:
                    colors.append((0, 255, 0))  # Green for low speed

        if opt.save_txt:
            result_json = json.loads(result.tojson())
            for idx, label in enumerate(labels):
                result_json['boxes'][idx]['label'] = label
            yield json.dumps({'results': result_json})
        else:
            im0 = result.plot()
            for idx, label in enumerate(labels):
                box = result.boxes[idx]
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 使用 xyxy 属性获取坐标
                color = colors[idx]  # Get the color for this label
                cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)  # Draw rectangle with the appropriate color
            im0 = cv2.imencode('.jpg', im0)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')


@app.route('/')
def index():
    """
    Video streaming home page.
    """
    return render_template('home.html')


# In flask applicatioin
@app.route('/model')
def model_page():
    """
    Display the model page and execute the corresponding Flask function.
    """
    return render_template('model.html')


@app.route('/predict', methods=['GET', 'POST'])
def video_feed():
    if request.method == 'POST':
        uploaded_file = request.files.get('myfile')
        save_txt = request.form.get('save_txt', 'F')  # Default to 'F' if save_txt is not provided

        if uploaded_file:
            source = Path(__file__).parent / 'raw_data' / uploaded_file.filename
            uploaded_file.save(source)
            opt.source = source
        else:
            opt.source, _ = update_options(request)

        opt.save_txt = True if save_txt == 'T' else False

    elif request.method == 'GET':
        opt.source, opt.save_txt = update_options(request)

    return Response(predict(opt), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '--weights', type=str, default='weights/guo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images',
                        help='source directory for images or videos')
    parser.add_argument('--conf', '--conf-thres', type=float, default=0.25,
                        help='object confidence threshold for detection')
    parser.add_argument('--iou', '--iou-thres', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='image size as scalar or (h, w) list, i.e. (640, 480)')
    parser.add_argument('--half', action='store_true', help='use half precision (FP16)')
    parser.add_argument('--device', default='', help='device to run on, i.e. cuda device=0/1/2/3 or device=cpu')
    parser.add_argument('--show', '--view-img', default=False, action='store_true', help='show results if possible')
    parser.add_argument('--save', action='store_true', help='save images with results')
    parser.add_argument('--save_txt', '--save-txt', action='store_true', help='save results as .txt file')
    parser.add_argument('--save_conf', '--save-conf', action='store_true', help='save results with confidence scores')
    parser.add_argument('--save_crop', '--save-crop', action='store_true', help='save cropped images with results')
    parser.add_argument('--show_labels', '--show-labels', default=True, action='store_true', help='show labels')
    parser.add_argument('--show_conf', '--show-conf', default=True, action='store_true', help='show confidence scores')
    parser.add_argument('--max_det', '--max-det', type=int, default=300, help='maximum number of detections per image')
    parser.add_argument('--vid_stride', '--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--stream_buffer', '--stream-buffer', default=False, action='store_true',
                        help='buffer all streaming frames (True) or return the most recent frame (False)')
    parser.add_argument('--line_width', '--line-thickness', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize model features')
    parser.add_argument('--augment', default=False, action='store_true',
                        help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', '--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--retina_masks', '--retina-masks', default=False, action='store_true',
                        help='whether to plot masks in native resolution')
    parser.add_argument('--classes', type=list,
                        help='filter results by class, i.e. classes=0, or classes=[0,2,3]')  # 'filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--boxes', default=True, action='store_false', help='Show boxes in segmentation predictions')
    parser.add_argument('--exist_ok', '--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--raw_data', '--raw-data', default=ROOT / 'data/raw', help='save raw images to data/raw')
    parser.add_argument('--port', default=4000, type=int, help='port deployment')
    opt, unknown = parser.parse_known_args()

    # print used arguments
    print_args(vars(opt))

    # Get port to deploy
    port = opt.port
    delattr(opt, 'port')

    # Create path for raw data
    raw_data = Path(opt.raw_data)
    raw_data.mkdir(parents=True, exist_ok=True)
    delattr(opt, 'raw_data')

    # Load model (Ensemble is not supported)
    model = YOLO(str(opt.model))

    # Run app
    app.run(host='0.0.0.0', port=port,
            debug=False)  # Don't use debug=True, model will be loaded twice (https://stackoverflow.com/questions/26958952/python-program-seems-to-be-running-twice)
