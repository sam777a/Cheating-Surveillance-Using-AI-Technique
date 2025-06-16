import cv2
import numpy as np
from ultralytics import YOLO
import logging
from utils import validate_frame

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load YOLO model
model = YOLO('yolov8n.pt')

def process_mobile_detection(frame):
    """Process frame for mobile detection using YOLO with stricter filtering."""
    try:
        frame = validate_frame(frame)
    except ValueError as e:
        logging.error(str(e))
        return frame, False

    # Perform YOLO detection
    results = model(frame)
    mobile_detected = False

    # Process detection results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            label = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            logging.debug(f"Detected object: {label}, confidence: {confidence:.2f}, bbox: ({x1}, {y1}, {x2}, {y2})")

            # Filter 1: Only consider "cell phone" (class ID 67 in COCO dataset)
            if class_id != 67:  # "cell phone" class ID in COCO
                logging.debug(f"Ignoring non-mobile object: {label}")
                continue

            # Filter 2: Higher confidence threshold to reduce false positives
            if confidence < 0.7:  # Increased from default (typically 0.5)
                logging.debug(f"Low confidence for mobile: {confidence:.2f}")
                continue

            # Filter 3: Aspect ratio check (mobile phones typically have width/height between 0.3 and 0.7)
            width = x2 - x1
            height = y2 - y1
            if width == 0 or height == 0:
                logging.debug("Invalid bounding box dimensions")
                continue
            aspect_ratio = width / height
            if not (0.3 <= aspect_ratio <= 0.7 or 1.4 <= aspect_ratio <= 3.3):  # Covers portrait and landscape
                logging.debug(f"Invalid aspect ratio for mobile: {aspect_ratio:.2f}")
                continue

            # If all filters pass, mark as mobile detected
            mobile_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Mobile: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            logging.info(f"Mobile phone detected: confidence={confidence:.2f}, bbox=({x1}, {y1}, {x2}, {y2})")
            break  # Stop after detecting the first valid mobile phone

    return frame, mobile_detected