import argparse
import time
from pathlib import Path

import cv2
import serial
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-based person tracker with serial output by default.")
    parser.add_argument("--source", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"), help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IOU threshold")
    parser.add_argument("--max-dist", type=float, default=50.0, help="Maximum centroid distance for tracking")
    parser.add_argument("--reset-interval", type=float, default=2.0, help="Time (s) to reset tracking")
    parser.add_argument("--serial", action="store_true", default=True,
                        help="Enable serial output (default: enabled)")
    parser.add_argument("--port", type=str, default="COM4",
                        help="Serial port (default: COM4)")
    return parser.parse_args()


class CentroidTracker:
    def __init__(self, max_distance: float = 50.0):
        self.next_id = 0
        self.objects: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # id -> (centroid, bbox)
        self.max_distance = max_distance

    def update(self, detections: list[tuple[np.ndarray, np.ndarray]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Assigns detections to existing objects or creates new IDs.
        detections: list of (centroid, bbox) where centroid = [x, y], bbox = [x1,y1,x2,y2]
        """
        # If no detections, keep existing objects
        if not detections:
            return self.objects

        # Initialize if no existing objects
        if not self.objects:
            for centroid, bbox in detections:
                self.objects[self.next_id] = (centroid, bbox)
                self.next_id += 1
            return self.objects

        # Prepare centroid arrays
        old_ids = list(self.objects.keys())
        old_centroids = np.stack([c for c, _ in self.objects.values()])
        new_centroids = np.stack([c for c, _ in detections])
        new_bboxes = [b for _, b in detections]

        # Compute pairwise distances and solve assignment
        dist_matrix = cdist(old_centroids, new_centroids)
        row_idx, col_idx = linear_sum_assignment(dist_matrix)

        assigned = set()
        updated_objects: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Match existing objects within threshold
        for r, c in zip(row_idx, col_idx):
            if dist_matrix[r, c] < self.max_distance:
                obj_id = old_ids[r]
                updated_objects[obj_id] = (new_centroids[c], new_bboxes[c])
                assigned.add(c)

        # Add unmatched detections as new objects
        for i, (centroid, bbox) in enumerate(detections):
            if i not in assigned:
                updated_objects[self.next_id] = (centroid, bbox)
                self.next_id += 1

        self.objects = updated_objects
        return self.objects


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.source}")

    model = YOLO(str(args.model))
    tracker = CentroidTracker(max_distance=args.max_dist)

    # Initialize serial by default
    ser = None
    if args.serial:
        ser = serial.Serial(args.port, 9600, timeout=1)
        time.sleep(2)

    last_reset = time.time()
    tracked_id = None
    last_seen = time.time()

    print("Press 'q', 'Q', or Esc to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Periodic reset
        if time.time() - last_reset > args.reset_interval:
            tracked_id = None
            last_reset = time.time()

        # Run detection
        results = model(frame, conf=args.conf, iou=args.iou, verbose=False)
        detections = []
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                bbox = np.array([int(x1), int(y1), int(x2), int(y2)])
                detections.append((centroid, bbox))

        objects = tracker.update(detections)

        # Track the first detected if none tracked
        if tracked_id is None and objects:
            tracked_id = next(iter(objects))

        # Draw tracked object
        if tracked_id in objects:
            centroid, bbox = objects[tracked_id]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tracked_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            last_seen = time.time()

            # Normalize and output
            u = float(centroid[0] / frame.shape[1])
            v = float(centroid[1] / frame.shape[0])
            print(f"UV: [{u:.2f}, {v:.2f}]", end="\r", flush=True)
            if ser:
                if u > 0.65:
                    ser.write(b"0")
                elif u < 0.35:
                    ser.write(b"2")
                else:
                    ser.write(b"1")
        
        # Reset if lost
        elif tracked_id is not None and time.time() - last_seen > 15:
            tracked_id = None

        # Display and handle exit keys
        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1)
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()


if __name__ == "__main__":
    main()