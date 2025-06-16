import cv2
import time
import os
import logging
import winsound
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from mobile_detection import process_mobile_detection
from utils import validate_frame
from flask import Flask, render_template, send_from_directory, Response
from flask_socketio import SocketIO
from threading import Lock
import threading
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

# Set up logging with timestamp
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure logs are written immediately
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# Flask app setup
app = Flask(__name__)
socketio = SocketIO(app)
thread = None
thread_lock = Lock()
camera_lock = Lock()  # Added for thread-safe camera access

# Global variable to store the current frame for streaming
current_frame = None

# Video capture (will be initialized in main())
cap = None

# Flag to control local display
DISPLAY_LOCAL = True  # Changed to True to enable local display

def initialize_camera():
    """Initialize webcam with enhanced debugging."""
    global cap
    cap = cv2.VideoCapture(0)  # Try index 0 (default webcam)
    if not cap.isOpened():
        # Try other indices if index 0 fails
        for index in range(1, 3):  # Try indices 1 and 2
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                print(f"Camera opened successfully on index {index}!")
                logging.info(f"Camera opened successfully on index {index}!")
                break
        if not cap.isOpened():
            logging.error("Could not open webcam after trying multiple indices")
            print("Error: Could not open webcam after trying multiple indices")
            raise RuntimeError("Error: Could not open webcam. Check if the webcam is connected and not in use.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    print("Camera opened successfully!")  # Debug confirmation
    logging.info(f"Camera initialized - Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    # Test reading a frame immediately after opening
    ret, frame = cap.read()
    if not ret or frame is None:
        logging.error("Failed to read test frame immediately after opening camera")
        print("Error: Failed to read test frame immediately after opening camera")
        raise RuntimeError("Error: Camera opened but cannot read frames.")
    print("Test frame read successfully: shape=%s" % str(frame.shape))
    return cap

def create_log_directory():
    """Create log directory for screenshots."""
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def draw_alert_border(frame, color=(0, 0, 255), thickness=10):
    """Draw a colored border around the frame for visual alerts."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w-1, h-1), color, thickness)
    return frame

def generate_frames():
    """Generate video frames with detection results for streaming and optional local display."""
    global current_frame, cap
    print("Starting generate_frames()...")  # Debug start of generator

    # Initialize system components
    log_dir = create_log_directory()

    # Calibration for head pose
    calibrated_angles = None
    start_time = time.time()
    calibration_success = False
    face_detection_attempts = 0
    MAX_ATTEMPTS = 10

    # Calibration for gaze
    gaze_calibrated = False
    gaze_calibration_start = None
    gaze_calibration_attempts = 0

    # Timers for each functionality
    head_misalignment_start_time = None
    eye_misalignment_start_time = None
    mobile_detection_start_time = None

    # Previous states
    previous_head_state = "Looking Center"
    previous_eye_state = "Looking Center"
    previous_mobile_state = False

    # Frame counter for optimizing mobile detection
    frame_count = 0
    mobile_detected = False

    # Alert flags and timing
    alert_triggered = False
    last_alert_time = 0
    ALERT_COOLDOWN = 2

    # Gaze detection failure tracking
    gaze_center_frames = 0
    GAZE_CENTER_THRESHOLD = 300  # ~5 seconds at 60 FPS

    frame_iteration = 0
    while True:
        frame_iteration += 1
        print(f"Frame iteration {frame_iteration}...")  # Debug each iteration
        with camera_lock:
            ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            logging.error("Failed to capture a valid frame from webcam")
            print("Error: Failed to capture a valid frame from webcam. ret=%s, frame=%s, size=%s" % (ret, frame, frame.size if frame is not None else "None"))
            break

        try:
            frame = validate_frame(frame)
            print("Frame captured successfully: shape=%s" % str(frame.shape))  # Confirm frame capture
        except ValueError as e:
            logging.error(f"Frame validation failed: {str(e)}")
            print(f"Frame validation failed: {str(e)}")
            break

        logging.debug(f"Main - Frame shape: {frame.shape}, dtype: {frame.dtype}, contiguous: {frame.flags['C_CONTIGUOUS']}")

        # Process head pose
        head_direction = "Looking Center"
        try:
            if time.time() - start_time <= 5:
                cv2.putText(frame, "Calibrating Head... Look straight at the camera", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                try:
                    _, result = process_head_pose(frame, None)
                    if isinstance(result, tuple):
                        calibrated_angles = result
                        calibration_success = True
                        gaze_calibration_start = time.time()
                        logging.info("Head pose calibration successful")
                        winsound.Beep(1000, 200)
                    head_direction = "Calibrating..."
                except Exception as e:
                    logging.error(f"Error during head pose calibration: {str(e)}")
                    print(f"Error during head pose calibration: {str(e)}")
                    head_direction = "Calibration Error"
            else:
                if not calibration_success:
                    face_detection_attempts += 1
                    if face_detection_attempts >= MAX_ATTEMPTS:
                        cv2.putText(frame, "Head calibration failed. Adjust lighting/position and press 'r'", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        head_direction = "Calibration Failed"
                    else:
                        head_direction = "Retrying Calibration..."
                else:
                    try:
                        frame, head_direction = process_head_pose(frame, calibrated_angles)
                        color = (0, 255, 0) if head_direction == "Looking Center" else (0, 0, 255)
                        cv2.putText(frame, f"Head Direction: {head_direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except Exception as e:
                        logging.error(f"Error in head pose processing: {str(e)}")
                        print(f"Error in head pose processing: {str(e)}")
                        head_direction = "Head Pose Detection Failed"
                        cv2.putText(frame, f"Head Direction: {head_direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        except Exception as e:
            logging.error(f"Unexpected error in head pose block: {str(e)}")
            print(f"Unexpected error in head pose block: {str(e)}")
            head_direction = "Head Pose Detection Failed"
            cv2.putText(frame, f"Head Direction: {head_direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Process gaze calibration and eye movement
        gaze_direction = "Looking Center"
        try:
            if calibration_success and not gaze_calibrated:
                if time.time() - gaze_calibration_start <= 5:
                    cv2.putText(frame, "Calibrating Gaze... Look straight at the camera", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    try:
                        frame, gaze_result = process_eye_movement(frame, calibrate=True)
                        if gaze_result == "Calibration Done":
                            gaze_calibrated = True
                            winsound.Beep(1000, 200)
                        gaze_direction = gaze_result
                    except Exception as e:
                        logging.error(f"Error during gaze calibration: {str(e)}")
                        print(f"Error during gaze calibration: {str(e)}")
                        gaze_direction = "Gaze Calibration Error"
                else:
                    gaze_calibration_attempts += 1
                    if gaze_calibration_attempts >= MAX_ATTEMPTS:
                        cv2.putText(frame, "Gaze calibration failed. Adjust lighting/position and press 'r'", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        gaze_direction = "Gaze Calibration Failed"
                    else:
                        gaze_direction = "Retrying Gaze Calibration..."
            else:
                try:
                    frame, gaze_direction = process_eye_movement(frame, calibrate=False)
                    color = (0, 255, 0) if gaze_direction == "Looking Center" else (0, 0, 255)
                    cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Track if gaze is stuck on "Looking Center"
                    if gaze_direction == "Looking Center":
                        gaze_center_frames += 1
                    else:
                        gaze_center_frames = 0

                    if gaze_center_frames > GAZE_CENTER_THRESHOLD:
                        cv2.putText(frame, "Gaze stuck? Adjust lighting/position", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except Exception as e:
                    logging.error(f"Error in eye movement processing: {str(e)}")
                    print(f"Error in eye movement processing: {str(e)}")
                    gaze_direction = "Eye Detection Failed"
                    cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        except Exception as e:
            logging.error(f"Unexpected error in eye movement block: {str(e)}")
            print(f"Unexpected error in eye movement block: {str(e)}")
            gaze_direction = "Eye Detection Failed"
            cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Process mobile detection
        frame_count += 1
        if frame_count % 10 == 0:
            try:
                frame, mobile_detected = process_mobile_detection(frame)
            except Exception as e:
                logging.error(f"Error in mobile detection: {str(e)}")
                print(f"Error in mobile detection: {str(e)}")
                mobile_detected = False
        color = (0, 255, 0) if not mobile_detected else (0, 0, 255)
        cv2.putText(frame, f"Mobile Detected: {mobile_detected}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Check for head misalignment
        alert_message = None
        current_time = time.time()
        if head_direction not in ["Looking Center", "No Face Detected", "Face Detection Failed", "Head Pose Detection Failed", "Calibrating...", "Calibration Failed", "Retrying Calibration..."]:
            if head_misalignment_start_time is None:
                head_misalignment_start_time = current_time
            elif current_time - head_misalignment_start_time >= 2:
                if current_time - last_alert_time >= ALERT_COOLDOWN:
                    filename = os.path.join(log_dir, f"head_{head_direction}_{int(current_time)}.png")
                    cv2.imwrite(filename, frame)
                    logging.info(f"Screenshot saved: {filename}")
                    print(f"Screenshot saved: {filename}")
                    alert_message = f"Alert: Head Misalignment - {head_direction}"
                    last_alert_time = current_time
                    socketio.emit('alert', {'message': alert_message})  # Emit alert to dashboard
                head_misalignment_start_time = None
        else:
            head_misalignment_start_time = None

        # Check for eye misalignment
        if gaze_direction not in ["Looking Center", "No Face Detected", "Face Detection Failed", "Eye Detection Failed", "Gaze Not Calibrated", "Calibrating Gaze...", "Calibration Done", "Gaze Calibration Failed", "Retrying Gaze Calibration..."]:
            if eye_misalignment_start_time is None:
                eye_misalignment_start_time = current_time
            elif current_time - eye_misalignment_start_time >= 2:
                if current_time - last_alert_time >= ALERT_COOLDOWN:
                    filename = os.path.join(log_dir, f"eye_{gaze_direction}_{int(current_time)}.png")
                    cv2.imwrite(filename, frame)
                    logging.info(f"Screenshot saved: {filename}")
                    print(f"Screenshot saved: {filename}")
                    alert_message = f"Alert: Eye Misalignment - {gaze_direction}"
                    last_alert_time = current_time
                    socketio.emit('alert', {'message': alert_message})  # Emit alert to dashboard
                eye_misalignment_start_time = None
        else:
            eye_misalignment_start_time = None

        # Check for mobile detection
        if mobile_detected:
            if mobile_detection_start_time is None:
                mobile_detection_start_time = current_time
            elif current_time - mobile_detection_start_time >= 2:
                if current_time - last_alert_time >= ALERT_COOLDOWN:
                    filename = os.path.join(log_dir, f"mobile_detected_{int(current_time)}.png")
                    cv2.imwrite(filename, frame)
                    logging.info(f"Screenshot saved: {filename}")
                    print(f"Screenshot saved: {filename}")
                    alert_message = "Alert: Mobile Phone Detected"
                    last_alert_time = current_time
                    socketio.emit('alert', {'message': alert_message})  # Emit alert to dashboard
                mobile_detection_start_time = None
        else:
            mobile_detection_start_time = None

        # Handle alerts
        if alert_message:
            if not alert_triggered:
                winsound.Beep(1500, 300)
                alert_triggered = True
                logging.warning(f"Alert triggered: {alert_message}")
            cv2.putText(frame, alert_message, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if frame_count % 8 < 4:
                frame = draw_alert_border(frame, color=(0, 0, 255))
        else:
            alert_triggered = False

        # Display instructions
        cv2.putText(frame, "Press 'r' to recalibrate, 'q' to quit", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the combined output locally if enabled
        if DISPLAY_LOCAL:
            cv2.imshow("Combined Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Local display closed by user (pressed 'q')")
                break
            elif key == ord('r'):
                calibrated_angles = None
                start_time = time.time()
                calibration_success = False
                gaze_calibrated = False
                gaze_calibration_start = None
                gaze_calibration_attempts = 0
                gaze_center_frames = 0
                logging.info("Recalibration triggered")

        # Encode the frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Failed to encode frame as JPEG")
            print("Error: Failed to encode frame as JPEG")
            break
        frame_bytes = buffer.tobytes()

        # Update the global frame for streaming
        with thread_lock:
            current_frame = frame_bytes

        print("Yielding frame for streaming...")  # Debug before yielding
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the invigilator dashboard."""
    print("Accessing / route")  # Add this for debugging
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering index.html: {str(e)}")
        logging.error(f"Error rendering index.html: {str(e)}")
        return "Error: Could not render dashboard. Check logs for details.", 500

@app.route('/video_feed')
def video_feed():
    """Stream the video feed to the dashboard."""
    print("Accessing /video_feed route")  # Add this for debugging
    try:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {str(e)}")
        logging.error(f"Error in video_feed: {str(e)}")
        return "Error: Could not stream video feed. Check logs for details.", 500

@app.route('/logs')
def list_logs():
    """Return a list of logged screenshots."""
    print("Accessing /logs route")  # Add this for debugging
    log_dir = 'log'
    try:
        files = os.listdir(log_dir)
        files = [f for f in files if f.endswith('.png')]
        return render_template('logs.html', files=files)
    except Exception as e:
        print(f"Error in list_logs: {str(e)}")
        logging.error(f"Error in list_logs: {str(e)}")
        return "Error: Could not list logs. Check logs for details.", 500

@app.route('/log/<filename>')
def serve_log(filename):
    """Serve a specific log file (screenshot)."""
    print(f"Accessing /log/{filename} route")  # Add this for debugging
    try:
        return send_from_directory('log', filename)
    except Exception as e:
        print(f"Error in serve_log: {str(e)}")
        logging.error(f"Error in serve_log: {str(e)}")
        return "Error: Could not serve log file. Check logs for details.", 500

def background_thread():
    """Background thread to emit alerts via SocketIO."""
    print("Starting SocketIO background thread...")  # Debug
    while True:
        socketio.sleep(1)

@socketio.on('connect')
def handle_connect():
    """Start the background thread when a client connects."""
    print("SocketIO client connected")  # Add this for debugging
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)

def main():
    """Main function to initialize the system and start the Flask server."""
    global cap
    try:
        # Initialize the camera
        initialize_camera()

        # Start the Flask server directly in the main thread
        print("Starting Flask server on http://127.0.0.1:5000...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, log_output=True, use_reloader=False)  # Disable reloader to avoid issues
        print("Flask server has stopped.")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(f"Error in main: {str(e)}")
    finally:
        if cap is not None:
            print("Releasing camera...")
            cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed.")

if __name__ == "__main__":
    main()