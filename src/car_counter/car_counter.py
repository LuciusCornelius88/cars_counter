import shutil
import sys
import math
import cvzone
import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from sort import Sort

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


def delecte_cache(input_path):
    for path in input_path.iterdir():
        if path.is_dir() and path.name == '__pycache__':
            shutil.rmtree(path)
        elif path.is_dir():
            delecte_cache(path)


def render_boxes(img, results, detections):
    for res in results:
        boxes = res.boxes
        for box in boxes:
            x1, y1, x2, y2 = (int(item) for item in box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = CLASS_NAMES[cls]

            if current_class in ["bicycle", "car", "motorbike", "bus", "truck"] and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=3, t=4, rt=1)
                # cvzone.putTextRect(img, f'{CLASS_NAMES[cls]} {conf}',
                #                    (max(0, x1), max(35, y1)), scale=1, thickness=1)

                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    return img, detections


def track_objects(img, tracker_results, limits, total_count):
    for res in tracker_results:
        x1, y1, x2, y2, id = (int(item) for item in res)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if total_count.count(id) == 0:
                total_count.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=1)

    return img, total_count


def main():
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 1280)
    # cap.set(4, 720)

    model = YOLO(f'{MODEL_PATH}yolov8n.pt')
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    cap = cv2.VideoCapture(f'{VIDEO_PATH}cars.mp4')
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask = cv2.imread(f'{MASKS_PATH}cars_mask.png')
    mask = cv2.resize(mask, (video_width, video_height), cv2.INTER_CUBIC)

    total_count = []
    limits = [400, 297, 673, 297]

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img_region = cv2.bitwise_and(img, mask)
        # img = cv2.flip(img, 180)
        results = model(img_region, stream=True)
        detections = np.empty((0, 5))

        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 5)

        img, detections = render_boxes(img, results, detections)
        tracker_results = tracker.update(detections)
        img, total_count = track_objects(img, tracker_results, limits, total_count)

        cvzone.putTextRect(img, f'Total cars: {len(total_count)}', (30, 30), scale=2, thickness=2)
        cv2.imshow('Video', img)

        if cv2.waitKey(1) == ord(' '):
            break

    # print(len(list(set(total_count))))

    cap.release()
    cv2.destroyAllWindows()
    delecte_cache(Path(__file__).parent.parent)


if __name__ == '__main__':
    main()
