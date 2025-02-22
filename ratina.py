import cv2
import mediapipe as mp
import time
import numpy as np

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

mp_face_mesh = mp.solutions.face_mesh# type: ignore

LEFT_EYE_CENTER = 468  # Approximate central point of left eye
RIGHT_EYE_CENTER = 473  # Approximate central point of right eye
NOSE_CENTER = 1  # Nose tip

def ratina_distance(video_path):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    start_time = time.time()
    total_ratio = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark])
                
                left_eye_center = landmarks[LEFT_EYE_CENTER]
                right_eye_center = landmarks[RIGHT_EYE_CENTER]
                nose_center = landmarks[NOSE_CENTER]
                
                left_eye_distance = calculate_distance(left_eye_center, nose_center)
                right_eye_distance = calculate_distance(right_eye_center, nose_center)
                
                ratio = left_eye_distance / right_eye_distance if right_eye_distance != 0 else 0
                total_ratio += ratio
                frame_count += 1
                
                # Draw lines and points on frame
                cv2.line(frame, tuple(left_eye_center.astype(int)), tuple(nose_center.astype(int)), (255, 0, 0), 2)
                cv2.line(frame, tuple(right_eye_center.astype(int)), tuple(nose_center.astype(int)), (0, 255, 0), 2)
                cv2.circle(frame, tuple(left_eye_center.astype(int)), 3, (255, 0, 0), -1)
                cv2.circle(frame, tuple(right_eye_center.astype(int)), 3, (0, 255, 0), -1)
                cv2.circle(frame, tuple(nose_center.astype(int)), 3, (0, 0, 255), -1)
                
                # Display distance ratios
                cv2.putText(frame, f"Ratio: {ratio:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Face Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    avg_ratio = (total_ratio / frame_count) if frame_count > 0 else 0
    return avg_ratio

if __name__ == "__main__":
    video_path = "video.mp4"  # Replace with actual video file path
    avg_ratio = ratina_distance(video_path)
    print(f"Average Distance Ratio per Minute: {avg_ratio:.2f}")
