# Pushup Form Coach – Computer Vision Project

## Overview

This project is a simple **Pushup Form Coach** built with computer vision.  
It uses my laptop webcam and a pose detection model to:

- Draw a skeleton on my body in real time  
- Track my **right elbow angle** during pushups  
- **Count my reps** automatically  
- Give basic feedback like **“Good depth”** or **“Too shallow”**  
- Record the whole session to a video file  


## Requirements

To run this project, you need Python and these libraries:

- `opencv-python`
- `mediapipe`
- `numpy`

You can install them with:

```bash
pip install opencv-python mediapipe numpy
```
## How to Run
- Open a terminal in this project folder.
- Run: python main.py
- A window will open showing your webcam.
- Position the laptop so the camera has a side view of you doing pushups.
- Do some pushups in front of the camera.
- Press q to quit and close the window.

## What you will see
On the webcam window you should see:
 - A pose skeleton drawn on your body
 - A text line showing: Reps: <number>
 - A text line showing the current elbow angle in degrees
 - A text line for Feedback:
     **Good depth** if you went low enough; 
     **Too shallow** if the pushup was not deep enough
 - The program also saves a video file in the same folder with a name like:
    pushup_session_YYYY-MM-DD_HH-MM-SS.avi
- This video includes the skeleton, rep count, and feedback overlay.

## Notes
 - The pushup counting is based on the right elbow angle.
 - If the counting is off, you can adjust the angle thresholds in main.py.
 - This project is meant as a simple demo of using pose estimation + geometry for exercise feedback.