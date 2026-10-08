import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import winsound

# 1. Saved model ko load karein
model = load_model("drowsiness_cnn_model.h5")
print("Model loaded successfully!")

# 2. MediaPipe Face Mesh initialize karein
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# MediaPipe Face Mesh landmark indices for eyes
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# 3. Web camera start karein
cap = cv2.VideoCapture(0)

IMG_SIZE = 32
closed_frame_count = 0
ALERT_THRESHOLD = 10

print("Drowsiness detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Process frame with MediaPipe
    results = face_mesh.process(rgb_frame)

    eyes_closed = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Helper function to get bounding box for an eye from landmarks
            def get_eye_roi(eye_indices):
                x_coords = [int(face_landmarks.landmark[i].x * w_frame) for i in eye_indices]
                y_coords = [int(face_landmarks.landmark[i].y * h_frame) for i in eye_indices]
                
                min_x, max_x = max(0, min(x_coords) - 5), min(w_frame, max(x_coords) + 5)
                min_y, max_y = max(0, min(y_coords) - 5), min(h_frame, max(y_coords) + 5)
                return min_x, min_y, max_x, max_y

            # Get stable eye coordinates
            rx1, ry1, rx2, ry2 = get_eye_roi(RIGHT_EYE_INDICES)
            lx1, ly1, lx2, ly2 = get_eye_roi(LEFT_EYE_INDICES)

            # Crop eyes from grayscale frame
            right_eye_img = gray[ry1:ry2, rx1:rx2]
            left_eye_img = gray[ly1:ly2, lx1:lx2]

            eye_scores = []
            
            # Predict for each eye if crops are valid
            for eye_img in [right_eye_img, left_eye_img]:
                if eye_img.size > 0:
                    try:
                        eye_resized = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
                        eye_normalized = eye_resized.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                        prediction = model.predict(eye_normalized, verbose=0)
                        eye_scores.append(prediction[0][0])
                    except Exception:
                        pass

            # If model predicted eyes as closed (score > 0.5)
            if eye_scores and all(score > 0.50 for score in eye_scores):
                eyes_closed = True

            # Draw eye bounding boxes (Red if closed, Green if open)
            box_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), box_color, 2)
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), box_color, 2)

    # Update closed frame counter logic
    if eyes_closed:
        closed_frame_count += 1
    else:
        closed_frame_count = max(0, closed_frame_count - 1)  # Smooth reset

    # Determine status & alert colors
    if closed_frame_count >= ALERT_THRESHOLD:
        status = "ALERT !! DRIVER IS DROWSY"
        color = (0, 0, 255)
        try :
            winsound.Beep(1000, 500)  # Beep sound for alert
        except:
            pass
    elif eyes_closed:
        status = "CLOSED!"
        color = (0, 0, 255)
    else:
        status = "Active"
        color = (0, 255, 0)

    # Display status on screen
    cv2.putText(frame, f"Status: {status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Closed Frames: {closed_frame_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()