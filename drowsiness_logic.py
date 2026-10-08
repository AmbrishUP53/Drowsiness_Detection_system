import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pygame

try:
    pygame.mixer.init()
    pygame.mixer.music.load("alertsound_wav.wav")
except Exception:
    pass

def load_detection_resources():
    model = load_model("drowsiness_cnn_model.h5")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    return model, face_mesh

RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def process_frame(frame, model, face_mesh, closed_frame_count, total_alerts):
    h_frame, w_frame, _ = frame.shape

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)

    results = face_mesh.process(rgb_frame)
    eyes_closed = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            def get_eye_roi(eye_indices):
                x_coords = [int(face_landmarks.landmark[i].x * w_frame) for i in eye_indices]
                y_coords = [int(face_landmarks.landmark[i].y * h_frame) for i in eye_indices]
                min_x, max_x = max(0, min(x_coords) - 5), min(w_frame, max(x_coords) + 5)
                min_y, max_y = max(0, min(y_coords) - 5), min(h_frame, max(y_coords) + 5)
                return min_x, min_y, max_x, max_y

            rx1, ry1, rx2, ry2 = get_eye_roi(RIGHT_EYE_INDICES)
            lx1, ly1, lx2, ly2 = get_eye_roi(LEFT_EYE_INDICES)

            right_eye_img = gray[ry1:ry2, rx1:rx2]
            left_eye_img = gray[ly1:ly2, lx1:lx2]

            eye_scores = []
            IMG_SIZE = 32
            for eye_img in [right_eye_img, left_eye_img]:
                if eye_img.size > 0:
                    try:
                        eye_resized = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
                        eye_normalized = eye_resized.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                        prediction = model.predict(eye_normalized, verbose=0)
                        eye_scores.append(prediction[0][0])
                    except Exception:
                        pass

            if eye_scores and all(score > 0.50 for score in eye_scores):
                eyes_closed = True

            box_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), box_color, 2)
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), box_color, 2)

    if eyes_closed:
        closed_frame_count += 1
    else:
        closed_frame_count = max(0, closed_frame_count - 1)

    ALERT_THRESHOLD = 10
    if closed_frame_count >= ALERT_THRESHOLD:
        status = "ALERT !! DRIVER IS DROWSY"
        color = (0, 0, 255)
        if closed_frame_count == ALERT_THRESHOLD:
            total_alerts += 1
        
        # Guaranteed looping sound using SND_ALIAS and SND_ASYNC | SND_LOOP
        try:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
        except Exception:
            print("there is error to playing the sound")
            
    elif eyes_closed:
        status = "CLOSED!"
        color = (0, 0, 255)
    else:
        status = "Active"
        color = (0, 255, 0)
        closed_frame_count = 0
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    cv2.putText(frame, f"Status: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Closed Frames: {closed_frame_count}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame, status, closed_frame_count, total_alerts