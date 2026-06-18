# EyesAI: Intelligent Visual Assistance System

## Project Overview

EyesAI is an innovative visual assistance system designed to empower visually impaired individuals by providing real-time audio descriptions of their surroundings. This project integrates advanced AI models with sensor technology to detect obstacles, identify key objects, and articulate environmental details, enhancing safety and independence.

![Screenshot](./images/gemini.jpg)
## Features

*   **LiDAR-based Obstacle Detection**: Utilizes a LiDAR sensor to accurately measure distances and detect immediate physical obstacles.
*   **YOLOv8 Object Recognition**: Employs the YOLOv8 model to identify crucial objects in the environment, such as traffic lights, stop signs, people, vehicles, and common pets.
*   **Gemini AI Contextual Description**: Leverages Google's Gemini AI to process visual and distance data, generating concise and informative audio descriptions of the scene.
*   **ElevenLabs Text-to-Speech**: Converts AI-generated descriptions into natural-sounding speech, delivered directly to the user.
*   **Raspberry Pi Camera Integration**: Seamlessly integrates with the Raspberry Pi camera (Picamera2) for real-time video capture.
*   **Intelligent Triggering**: Features a debouncing mechanism to prevent repetitive alerts and an auto-trigger function to provide updates during periods of inactivity.

![Screenshot](./images/yolo.jpg)
## Architecture Summary

EyesAI operates as an embedded application, primarily intended for deployment on a Raspberry Pi. The system continuously monitors the environment using a Picamera2 module for visual input and a LiDAR sensor for depth perception. Raw data is processed by a Python application: YOLOv8 performs object detection on camera frames, while Google Gemini AI synthesizes visual and distance information into a textual description. This description is then vocalized via ElevenLabs' text-to-speech service, providing auditory feedback to the user. The architecture is designed for real-time responsiveness and efficient resource utilization on embedded hardware.

![Screenshot](./images/tech.jpg)
## Tech Stack

*   **Core Language**: Python
*   **Computer Vision**: OpenCV (`cv2`), `ultralytics` (YOLOv8), `Picamera2`
*   **Artificial Intelligence**: `google-generativeai` (Gemini API)
*   **Text-to-Speech**: `requests` (for ElevenLabs API)
*   **Hardware Communication**: `pyserial` (for LiDAR)
*   **Audio Playback**: `subprocess` (for `mpg321`)
*   **Utilities**: `logging`, `time`, `signal`, `sys`, `PIL` (Pillow)
*   **Deployment Target**: Raspberry Pi, Linux-based systems

## Setup and Installation

### Prerequisites

*   Raspberry Pi (or similar embedded Linux device)
*   Raspberry Pi Camera Module (compatible with Picamera2)
*   LiDAR Sensor (e.g., TF-Luna, connected via UART)
*   External speakers or headphones
*   Python 3.x
*   `mpg321` audio player (install via `sudo apt-get install mpg321`)

### API Keys

You will need API keys for the following services:

*   **Google Gemini API**: Obtain from [Google AI Studio](https://aistudio.google.com/).
*   **ElevenLabs API**: Obtain from [ElevenLabs](https://elevenlabs.io/).

Replace the placeholder API keys in `app.py`:

```python
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
ELEVENLABS_API_KEY = "YOUR_ELEVENLABS_API_KEY"
```

### Dependencies

Install the required Python packages:

```bash
pip install opencv-python ultralytics google-generativeai requests pyserial Pillow picamera2
```

### Hardware Connections

*   **LiDAR**: Connect the LiDAR sensor to the Raspberry Pi's UART pins (e.g., `/dev/ttyAMA0`). Ensure proper power and ground connections.
*   **Camera**: Connect the Picamera2 module to the CSI port.

## Usage

To run the EyesAI system, execute the `app.py` script:

```bash
python app.py
```

The system will start, initialize the camera and LiDAR, and begin processing environmental data. Audio descriptions will be played through the connected audio output.

Press `Ctrl+C` to gracefully shut down the application.

## Configuration

Key configuration parameters can be adjusted at the beginning of `app.py`:

*   `LIDAR_PORT`: Serial port for the LiDAR sensor.
*   `BAUD_RATE`: Baud rate for LiDAR communication.
*   `MIN_DISTANCE`, `MAX_DISTANCE`: LiDAR distance thresholds.
*   `MIN_STRENGTH`: Minimum signal strength for LiDAR readings.
*   `DEBOUNCE_SECONDS`: Time to wait between triggers.
*   `TARGET_CLASSES`: List of objects YOLOv8 should specifically look for.
*   `VOICE_ID`: ElevenLabs voice ID.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is open-source. (Consider adding a specific license, e.g., MIT, Apache 2.0)