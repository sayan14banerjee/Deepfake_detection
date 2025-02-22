import cv2
import mediapipe as mp
import time
import numpy as np

def eye_aspect_ratio(landmarks, eye_indices):
    A = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    B = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    C = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    return (A + B) / (2.0 * C)

mp_face_mesh = mp.solutions.face_mesh # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
BLINK_THRESHOLD = 0.22
CONSECUTIVE_FRAMES = 3  # Number of frames eyes must be closed to count as a blink

def count_blinks(video_path):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    blink_count = 0
    blink_frame_counter = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark])
                ear = (eye_aspect_ratio(landmarks, LEFT_EYE) + eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2.0
                
                if ear < BLINK_THRESHOLD:
                    blink_frame_counter += 1
                else:
                    if blink_frame_counter >= CONSECUTIVE_FRAMES:
                        blink_count += 1
                    blink_frame_counter = 0
        
        cv2.putText(frame, f"Blinks: {blink_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Blink Detection", frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    blink_rate_per_min = (blink_count / total_time) * 60 if total_time > 0 else 0
    return blink_count, blink_rate_per_min

if __name__ == "__main__":
    video_path = "video.mp4"  # Replace with actual video file path
    total_blinks, blink_rate = count_blinks(video_path)
    print(f"Total Blinks: {total_blinks}")
    print(f"Blink Rate: {blink_rate:.2f} blinks per minute")
