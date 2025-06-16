import cv2
import mediapipe as mp
import numpy as np
import logging
from utils import validate_frame
from collections import deque

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enable iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Calibration storage
calibration_samples = deque(maxlen=30)
calibrated_offsets = None

def get_eye_region(frame, landmarks, eye_indices, h, w):
    """Extract eye region from landmarks."""
    eye_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices], dtype=np.int32)
    x, y, ew, eh = cv2.boundingRect(eye_points)
    expand = 3  # Further reduced for more sensitivity
    x = max(0, x - expand)
    y = max(0, y - expand)
    ew = ew + 2 * expand
    eh = eh + 2 * expand
    x_end = min(w, x + ew)
    y_end = min(h, y + eh)
    if x_end <= x or y_end <= y:
        logging.debug("Invalid eye region dimensions")
        return None, None
    return frame[y:y_end, x:x_end], (x, y, ew, eh)

def process_eye_movement(frame, calibrate=False):
    """Process eye movement and return frame with gaze direction."""
    global calibrated_offsets

    try:
        frame = validate_frame(frame)
    except ValueError as e:
        logging.error(str(e))
        return frame, str(e)

    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    gaze_direction = "Looking Center"
    if not results.multi_face_landmarks:
        logging.debug("No faces detected in eye_movement")
        return frame, "No Face Detected"

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = frame.shape

    # Left eye (landmarks 468-473 for iris, 33-41 for outer eye)
    left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173]
    left_iris_indices = list(range(468, 473))  # Use all iris landmarks
    left_eye, left_rect = get_eye_region(frame, face_landmarks.landmark, left_eye_indices, h, w)

    # Right eye (landmarks 473-478 for iris, 263-271 for outer eye)
    right_eye_indices = [263, 466, 388, 387, 386, 385, 384, 398]
    right_iris_indices = list(range(473, 478))
    right_eye, right_rect = get_eye_region(frame, face_landmarks.landmark, right_eye_indices, h, w)

    if left_rect:
        x, y, ew, eh = left_rect
        cv2.rectangle(frame, (x, y), (x + ew, y + eh), (0, 255, 0), 2)
    if right_rect:
        x, y, ew, eh = right_rect
        cv2.rectangle(frame, (x, y), (x + ew, y + eh), (0, 255, 0), 2)

    # Use iris landmarks for gaze
    if left_rect and right_rect:
        # Average multiple iris landmarks for better accuracy
        left_iris_positions = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in left_iris_indices]
        right_iris_positions = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in right_iris_indices]
        left_iris_x, left_iris_y = np.mean(left_iris_positions, axis=0)
        right_iris_x, right_iris_y = np.mean(right_iris_positions, axis=0)

        # Normalize iris position within eye region
        lx = (left_iris_x - left_rect[0]) / left_rect[2]
        ly = (left_iris_y - left_rect[1]) / left_rect[3]
        rx = (right_iris_x - right_rect[0]) / right_rect[2]
        ry = (right_iris_y - right_rect[1]) / right_rect[3]

        # Draw iris positions (using the center landmark for visualization)
        left_iris_pos = (int(left_iris_x), int(left_iris_y))
        right_iris_pos = (int(right_iris_x), int(right_iris_y))
        cv2.circle(frame, left_iris_pos, 5, (0, 0, 255), -1)
        cv2.circle(frame, right_iris_pos, 5, (0, 0, 255), -1)

        # Display normalized coordinates on-screen
        cv2.putText(frame, f"L Iris: ({lx:.2f}, {ly:.2f})", (left_rect[0], left_rect[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Calibration step with validation
        if calibrate:
            # Validate sample consistency (discard outliers)
            if len(calibration_samples) > 0:
                last_sample = calibration_samples[-1]
                if (abs(lx - last_sample[0]) > 0.1 or abs(ly - last_sample[1]) > 0.1 or
                    abs(rx - last_sample[2]) > 0.1 or abs(ry - last_sample[3]) > 0.1):
                    logging.debug(f"Discarding inconsistent sample: lx={lx:.2f}, ly={ly:.2f}, rx={rx:.2f}, ry={ry:.2f}")
                    return frame, "Calibrating Gaze..."
            calibration_samples.append((lx, ly, rx, ry))
            if len(calibration_samples) >= 20:
                avg_positions = np.mean(calibration_samples, axis=0)
                calibrated_offsets = tuple(avg_positions)
                logging.info(f"Gaze calibration completed: lx={avg_positions[0]:.2f}, ly={avg_positions[1]:.2f}, rx={avg_positions[2]:.2f}, ry={avg_positions[3]:.2f}")
                return frame, "Calibration Done"
            return frame, "Calibrating Gaze..."

        # Ensure calibration is complete before proceeding
        if calibrated_offsets is None:
            return frame, "Gaze Not Calibrated"

        # Adjust iris positions based on calibration
        lx_offset, ly_offset, rx_offset, ry_offset = calibrated_offsets
        lx_adjusted = lx - lx_offset
        ly_adjusted = ly - ly_offset
        rx_adjusted = rx - rx_offset
        ry_adjusted = ry - ry_offset

        # Display adjusted coordinates on-screen
        cv2.putText(frame, f"L Adj: ({lx_adjusted:.2f}, {ly_adjusted:.2f})", (left_rect[0], left_rect[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"R Adj: ({rx_adjusted:.2f}, {ry_adjusted:.2f})", (right_rect[0], right_rect[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        logging.debug(f"Adjusted Left iris: x={lx_adjusted:.2f}, y={ly_adjusted:.2f}")
        logging.debug(f"Adjusted Right iris: x={rx_adjusted:.2f}, y={ry_adjusted:.2f}")

        # Relaxed thresholds and modified logic for gaze direction
        if lx_adjusted < -0.1 or rx_adjusted < -0.1:  # Relaxed to 0.1, use OR
            gaze_direction = "Looking Left"
        elif lx_adjusted > 0.1 or rx_adjusted > 0.1:
            gaze_direction = "Looking Right"
        elif ly_adjusted < -0.1 or ry_adjusted < -0.1:
            gaze_direction = "Looking Up"
        elif ly_adjusted > 0.1 or ry_adjusted > 0.1:
            gaze_direction = "Looking Down"
        else:
            gaze_direction = "Looking Center"

    return frame, gaze_direction