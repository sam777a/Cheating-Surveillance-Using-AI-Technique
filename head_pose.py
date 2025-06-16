import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import logging
from utils import validate_frame

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3D Model Points (simplified for head pose estimation)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),   # Chin (landmark 152)
    (-225.0, 170.0, -135.0),# Left eye (landmark 33)
    (225.0, 170.0, -135.0), # Right eye (landmark 263)
    (-150.0, -150.0, -125.0),# Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0) # Right mouth corner (landmark 291)
], dtype=np.float64)

# Camera Calibration
focal_length = 640  # Adjusted for MediaPipe
center = (320, 240)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

# Smoothing and Calibration
ANGLE_HISTORY_SIZE = 10
yaw_history = deque(maxlen=ANGLE_HISTORY_SIZE)
pitch_history = deque(maxlen=ANGLE_HISTORY_SIZE)
roll_history = deque(maxlen=ANGLE_HISTORY_SIZE)
calibration_samples = deque(maxlen=30)

previous_state = "Looking Center"

def normalize_angle(angle):
    """Normalize angle to [-180, 180] degrees."""
    return ((angle + 180) % 360) - 180

def get_head_pose_angles(image_points):
    """Compute head pose angles (pitch, yaw, roll) in degrees."""
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            logging.warning("solvePnP failed to compute head pose angles")
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = 0

        pitch, yaw, roll = np.degrees(pitch), np.degrees(yaw), np.degrees(roll)
        pitch = normalize_angle(pitch)
        yaw = normalize_angle(yaw)
        roll = normalize_angle(roll)
        angles = (pitch, yaw, roll)

        logging.debug(f"Computed angles: pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}")
        return angles
    except Exception as e:
        logging.error(f"Error in get_head_pose_angles: {str(e)}")
        return None

def smooth_angle(angle_history, new_angle):
    """Smooth angles using a moving average."""
    angle_history.append(new_angle)
    return np.mean(angle_history)

def process_head_pose(frame, calibrated_angles=None):
    """Process head pose and return frame with head direction."""
    global previous_state

    try:
        frame = validate_frame(frame)
    except ValueError as e:
        logging.error(str(e))
        return frame, str(e)

    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:
        logging.debug("No faces detected in head_pose")
        return frame, "No Face Detected"

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = frame.shape
    image_points = np.array([
        (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),      # Nose tip
        (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # Chin
        (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),   # Left eye
        (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), # Right eye
        (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),   # Left mouth corner
        (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)  # Right mouth corner
    ], dtype=np.float64)

    angles = get_head_pose_angles(image_points)
    if angles is None:
        logging.debug("Failed to compute head pose angles")
        return frame, "Angle Computation Failed"

    pitch, yaw, roll = angles
    pitch = smooth_angle(pitch_history, pitch)
    yaw = smooth_angle(yaw_history, yaw)
    roll = smooth_angle(roll_history, roll)

    if calibrated_angles is None:
        calibration_samples.append((pitch, yaw, roll))
        if len(calibration_samples) >= 20:
            avg_angles = np.mean(calibration_samples, axis=0)
            logging.info(f"Calibration completed: pitch={avg_angles[0]:.2f}, yaw={avg_angles[1]:.2f}, roll={avg_angles[2]:.2f}")
            return frame, tuple(avg_angles)
        return frame, "Calibrating..."

    pitch_offset, yaw_offset, roll_offset = calibrated_angles
    PITCH_THRESHOLD = max(3, abs(pitch_offset) * 0.6)
    YAW_THRESHOLD = max(6, abs(yaw_offset) * 0.8)
    ROLL_THRESHOLD = max(3, abs(roll_offset) * 0.8)

    logging.debug(f"Angles after smoothing: pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}")
    logging.debug(f"Offsets: pitch_offset={pitch_offset:.2f}, yaw_offset={yaw_offset:.2f}, roll_offset={roll_offset:.2f}")
    logging.debug(f"Thresholds: pitch={PITCH_THRESHOLD:.2f}, yaw={YAW_THRESHOLD:.2f}, roll={ROLL_THRESHOLD:.2f}")

    if (abs(yaw - yaw_offset) <= YAW_THRESHOLD and 
        abs(pitch - pitch_offset) <= PITCH_THRESHOLD and 
        abs(roll - roll_offset) <= ROLL_THRESHOLD):
        current_state = "Looking Center"
    elif yaw < yaw_offset - YAW_THRESHOLD:
        current_state = "Looking Left"
    elif yaw > yaw_offset + YAW_THRESHOLD:
        current_state = "Looking Right"
    elif pitch > pitch_offset + PITCH_THRESHOLD:
        current_state = "Looking Up"
    elif pitch < pitch_offset - PITCH_THRESHOLD:
        current_state = "Looking Down"
    elif abs(roll - roll_offset) > ROLL_THRESHOLD:
        current_state = "Tilted"
    else:
        current_state = previous_state

    previous_state = current_state
    return frame, current_state