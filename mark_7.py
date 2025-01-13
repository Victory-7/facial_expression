import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

K = 1000
# Load the dataset
dataset_path = "facial_expression_data_new.csv"
data = pd.read_csv(dataset_path)

# Prepare data for training
X = data.iloc[:, :-1].values  # Landmark positions
y = data.iloc[:, -1].values   # Expressions

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=False, max_num_faces=1)

# Function to calculate distance based on the size of an object
def calculate_distance(area, K):
    if area > 0:
        return K / np.sqrt(area)
    else:
        return float('inf')

# Function to infer actions based on node movement
def infer_action(landmarks):
    if landmarks:
        left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        if left_hand.y < nose.y and right_hand.y < nose.y:
            return "Hands raised"
        elif left_hand.y > nose.y and right_hand.y > nose.y:
            return "Hands lowered"
        else:
            return "Neutral position"
    return "No action detected"

# Function to detect sitting, standing, or walking
def detect_movement(landmarks):
    if landmarks:
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Calculate angles at knees
        left_leg_angle = np.degrees(np.arctan2(left_hip.y - left_knee.y, left_hip.x - left_knee.x) -
                                     np.arctan2(left_ankle.y - left_knee.y, left_ankle.x - left_knee.x))
        right_leg_angle = np.degrees(np.arctan2(right_hip.y - right_knee.y, right_hip.x - right_knee.x) -
                                      np.arctan2(right_ankle.y - right_knee.y, right_ankle.x - right_knee.x))

        # Normalize angles to the range [0, 180]
        left_leg_angle = abs(left_leg_angle) % 180
        right_leg_angle = abs(right_leg_angle) % 180

        # Determine movement based on angles and relative positions
        if left_leg_angle > 160 and right_leg_angle > 160:
            return "Standing"
        elif left_leg_angle < 120 and right_leg_angle < 120:
            return "Sitting"
        else:
            return "Walking"
    return "Movement not detected"

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for pose estimation
        pose_results = pose.process(rgb_frame)

        # Process the frame for face mesh
        face_results = face_mesh.process(rgb_frame)

        # Annotate the frame with pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate distance based on the nose node
            nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            if nose.visibility > 0.5:  # Only consider visible landmarks
                area = 10000  # Example fixed area; adjust as needed
                distance = calculate_distance(area, K)
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Infer action
            action = infer_action(pose_results.pose_landmarks.landmark)
            cv2.putText(frame, f"Action: {action}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Detect movement
            movement = detect_movement(pose_results.pose_landmarks.landmark)
            cv2.putText(frame, f"Movement: {movement}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Detect facial expressions
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                # Predict expression
                expression = knn.predict([landmarks])[0]
                cv2.putText(frame, f"Expression: {expression}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Pose and Facial Expression Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
