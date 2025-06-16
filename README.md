# Cheating Surveillance Using AI Technique

## Overview

Proposed System / Method of solution
The proposed system is a cutting-edge, real-time AI-powered surveillance solution designed to detect cheating behaviors during online interviews and examinations, thereby enhancing academic integrity in virtual environments. It overcomes the limitations of traditional proctoring by leveraging advanced artificial intelligence and computer vision technologies to provide continuous, non-intrusive monitoring.
At the core of the system is the integration of powerful open-source libraries and frameworks. OpenCV is used for real-time video capture and image processing, while MediaPipe facilitates precise facial landmark detection critical for analyzing head and eye movements. Dlib enhances face detection robustness, and YOLOv12, a state-of-the-art object detection model, identifies prohibited objects such as mobile phones within the candidate’s environment.
The system continuously monitors key behavioral cues indicative of cheating, including suspicious head movements, frequent glances away from the screen, and the presence of unauthorized devices. Upon detecting such suspicious activity, the system immediately generates alerts to notify remote invigilators for timely intervention.
A distinguishing feature of the solution is its Web-Based Invigilator Dashboard, developed using Flask and Flask-SocketIO, which offers a real-time, browser-accessible interface. This dashboard allows invigilators to view live video streams, receive instant notifications of flagged behaviors, and review detailed logs containing timestamps and visual evidence such as screenshots. This facilitates both immediate response and post-examination review.
The modular and scalable Python-based architecture ensures smooth integration of various detection modules and supports adaptability across diverse examination scenarios, including academic institutions and professional certification environments. By combining AI-driven analytics with an accessible monitoring platform, the system effectively addresses concerns related to scalability, accuracy, and user privacy in remote proctoring.
Overall, this project aims to redefine online examination supervision by delivering a reliable, intelligent, and user-friendly solution that upholds academic honesty and fosters trust in digital assessment environments.

       The key contributions of the proposed system include:
•Head and Eye Movement Detection: Accurately identifies suspicious head turns and prolonged eye gaze away from the screen, which are common indicators of cheating during online exams and interviews.
•Unauthorized Device Detection: Detects the presence of prohibited devices such as mobile phones within the examination environment, using advanced object detection models to prevent unauthorized communication or information access.
•Real-Time Alert Generation: Provides immediate notifications to remote invigilators upon detecting suspicious behaviors or unauthorized device usage, enabling timely intervention.
•Web-Based Monitoring Dashboard: Offers a user-friendly, real-time dashboard for invigilators to remotely observe live video feeds, receive alerts, and review logged evidence, enhancing the effectiveness and scalability of online proctoring

2.5.1	Methodology
The operational workflow of the Cheating Surveillance System begins with the activation of the candidate’s camera to capture live video footage during online exams or interviews. The system continuously processes this video stream in real-time, utilizing AI techniques to analyze head movements, eye gaze, and the presence of unauthorized objects such as mobile phones.
Facial landmarks are detected using MediaPipe, enabling precise tracking of head and pupil movements to identify suspicious behaviors like looking away from the screen for prolonged periods. Simultaneously, the YOLOv12 model, combined with dlib, scans the environment for prohibited devices in the candidate’s vicinity.
When suspicious behavior or unauthorized device usage is detected, the system instantly generates alerts that are sent to remote invigilators through the web-based dashboard. This dashboard, built with Flask and Flask-SocketIO, allows invigilators to monitor live feeds, receive real-time notifications, and review evidence captured as screenshots or video clips.
By providing continuous, automated monitoring and immediate alerting, the system enhances the integrity and fairness of online assessments, enabling timely intervention and maintaining a secure examination environment.


## Technologies Used
- **Python**
- **OpenCV** (for video processing)
- MediaPipe (face/eye tracking), 
- **dlib** (for facial landmark detection)
- **YOLO (You Only Look Once)** (for object detection)
- PyTorch
- **Roboflow Dataset** (for training the mobile detection model)

## Folder Structure
```

├── .vscode/
├── Cheating-Surveillance-System-main/
│   ├── __pycache__/
│   ├── Demo_vid/                  # Folder for demo or test videos
│   ├── log/                       # Directory for saving screenshot logs
│   ├── model/                     # Contains pretrained models
│   │   ├── best_yolov8.pt
│   │   ├── best_yolov12.pt
│   │   └── shape_predictor_68_face_landmarks.dat
│   ├── templates/                 # HTML files for web dashboard
│   │   ├── index.html             # Dashboard UI
│   │   └── logs.html              # Logs viewer UI
│   ├── debug.log                  # System logs for debugging
│   ├── eye_movement.py            # Eye tracking and pupil movement detection
│   ├── head_pose.py               # Head direction detection using dlib
│   ├── main.py                    # Main control script (stream, detect, serve)
│   ├── mobile_detection.py        # YOLO-based mobile phone detection
│   ├── README.md                  # Project documentation
│   ├── requirements.txt           # Required Python packages
│   ├── requirements1.txt          # Possibly a backup or test dependency list
│   ├── utils.py                   # Utility/helper functions
│   ├── yolov8n.pt                 # Additional YOLOv8 model
├── dlib-env-310/                  # Virtual environment directory
├── log/             
```

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- OpenCV
- dlib
- torch (for YOLO)
- roboflow (for dataset access)

### Setup
1. Clone the repository:
   ```bash
   https://github.com/sam777a/Cheating-Surveillance-Using-AI-Technique.git
   cd Cheating-Surveillance-Using-AI-Technique
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the **Shape Predictor 68** model:
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```
4. Set up the YOLO model:  
   - You have trained your YOLO model on the [Roboflow Cellphone Dataset](https://universe.roboflow.com/d1156414/cellphone-0aodn).  
   - Download the trained YOLO weights and place the weights file in the `models/` directory.

## Usage
### Running the Surveillance System
To start real-time monitoring, run:
```bash
python main.py
```

### How It Works
1. **Facial Landmark Detection**: Detects and tracks head movements and pupil direction.
2. **YOLO-based Object Detection**: Identifies mobile phones in the video feed.
3. **Cheating Behavior Analysis**: Flags abnormal behavior such as frequent head turning or gaze shifts.

## Dataset
The mobile phone detection model is trained on the **Roboflow Cellphone Detection Dataset**. You can access it here: [Roboflow Cellphone Dataset](https://universe.roboflow.com/d1156414/cellphone-0aodn).

## Contributing
Feel free to submit issues and pull requests! If you have improvements or additional features, contribute by following these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature-branch`
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [dlib](http://dlib.net/)
- [OpenCV](https://opencv.org/)
- [YOLO](https://github.com/ultralytics/yolov5)
- [Roboflow](https://roboflow.com/) for dataset support
