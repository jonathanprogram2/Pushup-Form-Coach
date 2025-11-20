import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime


def calculate_angle(a, b, c):
    """
    Calculating the angle at point b (in degrees) given three points:
    a, b, c are x,y pairs.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # Computing cosine of angle using dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)

    # Clamping to valid range to avoid NaNs from numerical issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def run_pushup_coach():
    """
    Here we'll open up the webcam and draw the pose skeleton.
    Which will detect the pose using MediaPipe.
    it will then compute right elbow angle.
    count pushups based on angle thresholds
    and give feedback: 'Good depth' vs 'Too Shallow'.
    """

    # Thresholds *** These are to be tweaked
    DOWN_THRESHOLD = 90    # elbow angle smaller than this = "down" position
    UP_THRESHOLD = 160     # elbow angle larger than this = "up" position
    GOOD_DEPTH_THRESHOLD = 80    # smallest angle in a rep; smaller = deeper pushup

    rep_count = 0
    stage = "up" 
    min_angle_current_rep = 999.0
    last_feedback = "Do a pushup!"


    # Initialize MediaPipe Pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Open default webcam
    cap = cv2.VideoCapture(0)

    # Check if webcam opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Video recording set up
    # frame size from webcam
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # creating videowriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    
    # timestamp in the filename so each run is unique...
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"pushup_session_{timestamp}.avi"

    out = cv2.VideoWriter(
        output_filename,
        fourcc,
        20.0,
        (frame_width, frame_height),
    )
    
    print(f"Recording video to: {output_filename}")
    
    # Use Pose model in a context manager
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break

            # Flip frame horizontally for a mirror view
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Running pose detection
            results = pose.process(rgb_frame)

            elbow_angle = None

            # Drawing pose landmarks on the original BGR frame
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Using RIGHT arm 
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                # converting normalized coordinates (0-1) to pixel coords
                shoulder_xy = (right_shoulder.x * width, right_shoulder.y * height)
                elbow_xy = (right_elbow.x * width, right_elbow.y * height)
                wrist_xy = (right_wrist.x * width, right_wrist.y * height)

                # Computing elbow angle
                elbow_angle = calculate_angle(shoulder_xy, elbow_xy, wrist_xy)

                # ---- Rep counting logic -----
                if elbow_angle is not None:
                    # tracking minimum angle during the "down" phase for depth feedback
                    if stage == "down":
                        min_angle_current_rep = min(min_angle_current_rep, elbow_angle)

                    if elbow_angle < DOWN_THRESHOLD:
                        # In bottom part of pushup
                        stage = "down"

                    elif elbow_angle > UP_THRESHOLD:
                        # Back to top position
                        if stage == "down":
                            # the completion of a rep (down -> up)
                            rep_count += 1

                            # Depth feedback based on minimum angle reached
                            if min_angle_current_rep < GOOD_DEPTH_THRESHOLD:
                                last_feedback = "Good depth"
                            else:
                                last_feedback = "Too shallow"

                            # Reset for next rep
                            min_angle_current_rep = 999.0

                        stage = "up"

                # Drawing pose skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                # Optional tag- drawing a small circle at the elbow
                cv2.circle(
                    frame,
                    (int(elbow_xy[0]), int(elbow_xy[1])),
                    8,
                    (0, 255, 255),
                    -1,
                )

            # ---- The text overlays ------
            # Rep count
            cv2.putText(
                frame,
                f"Reps: {rep_count}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Current elbow angle
            if elbow_angle is not None:
                cv2.putText(
                   frame,
                   f"Elbow angle: {int(elbow_angle)} deg",
                   (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,
                   (255, 255, 0),
                   2,
                   cv2.LINE_AA,
                )

            # Feedback message ...
            cv2.putText(
                   frame,
                   f"Feedback: {last_feedback}",
                   (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,
                   (0, 255, 255),
                   2,
                   cv2.LINE_AA,
            )

            # Instructions....
            cv2.putText(
                frame,
                "Press 'q' to quit",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # writing frame to video file 
            out.write(frame)

            # Showing the frame
            cv2.imshow("Pushup Coach", frame)

            # Quit when the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording finished. Video saved.")

if __name__ == "__main__":
    run_pushup_coach()