import shutil
import sys
import math
import time
import cvzone
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sort import Sort


WINDOW_NAME = 'Window'
VIDEO_PATH = './data/videos/'
MODEL_PATH = './yolo_weights/'
MASKS_PATH = './data/masks/'
CLASS_NAMES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"
               ]


def delete_cache(input_path):
    for path in input_path.iterdir():
        if path.is_dir() and path.name == '__pycache__':
            shutil.rmtree(path)
        elif path.is_dir():
            delete_cache(path)


def track_mouse_pos(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        img = param.copy()
        cv2.putText(img, f'x: {x}, y: {y}', (30, 35),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)
        cv2.imshow(WINDOW_NAME, img)


def draw_line(img, active_time, limits, threshold=0.4):
    if time.time() - active_time >= threshold:
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    else:
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    return img


def render_boxes(img, results, detections):
    for res in results:
        for box in res.boxes:
            x1, y1, x2, y2 = (int(item) for item in box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=5, rt=2, colorR=(0, 0, 255))
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            current_class = CLASS_NAMES[cls]

            if current_class == 'person' and conf > 0.4:
                # cvzone.putTextRect(img, f'{current_class} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=1, thickness=1, colorR=(255, 50, 50))
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    return img, detections


def track_objects(img, tracker_results, limits_up, limits_down,
                  total_counts_up, total_counts_down, line_up_active_time,
                  line_down_active_time):
    for res in tracker_results:
        x1, y1, x2, y2, id = (int(item) for item in res)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        if limits_up[0] < cx < limits_up[2] and limits_up[1] + 20 > cy > limits_up[3] - 20:
            if total_counts_up.count(id) == 0:
                total_counts_up.append(id)
                line_up_active_time = time.time()
                cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 255, 0), 5)
        elif limits_down[0] < cx < limits_down[2] and limits_down[1] - 20 < cy < limits_down[3] + 20:
            if total_counts_down.count(id) == 0:
                total_counts_down.append(id)
                line_down_active_time = time.time()
                cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 255, 0), 5)

        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=1)

    return img, total_counts_up, total_counts_down, line_up_active_time, line_down_active_time


def main():
    model = YOLO(f'{MODEL_PATH}yolov8n.pt')
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    cap = cv2.VideoCapture(f'{VIDEO_PATH}people.mp4')
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask = cv2.imread(f'{MASKS_PATH}elevator_mask.png')
    mask = cv2.resize(mask, (video_width, video_height), cv2.INTER_CUBIC)

    cv2.namedWindow(WINDOW_NAME)

    total_counts_up = []
    total_counts_down = []
    limits_up = [110, 200, 360, 200]
    limits_down = [570, 570, 820, 570]

    line_up_active_time = 0
    line_down_active_time = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        frame = cv2.bitwise_and(img, mask)
        results = model(frame, stream=True)
        detections = np.empty((0, 5))

        img = draw_line(img, active_time=line_up_active_time, limits=limits_up)
        img = draw_line(img, active_time=line_down_active_time, limits=limits_down)

        img, detections = render_boxes(img, results, detections)
        tracker_results = tracker.update(detections)

        (img, total_counts_up, total_counts_down,
         line_up_active_time, line_down_active_time) = track_objects(img, tracker_results, limits_up,
                                                                     limits_down, total_counts_up,
                                                                     total_counts_down, line_up_active_time,
                                                                     line_down_active_time)

        cv2.setMouseCallback(WINDOW_NAME, track_mouse_pos, img)
        cvzone.putTextRect(img, f'Up: {len(total_counts_up)}', (1000, 200), scale=2, thickness=2)
        cvzone.putTextRect(img, f'Down: {len(total_counts_down)}', (1000, 260), scale=2, thickness=2)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key == ord(' '):
            break
        elif key == ord('p'):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    delete_cache(Path(__file__).parent.parent)
