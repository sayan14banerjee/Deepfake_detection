import cv2
import mediapipe as mp
import time
import numpy as np

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

mp_face_mesh = mp.solutions.face_mesh # type: ignore

# Lip corner landmarks
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14

def lip_movement(video_path):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
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
                
                left_lip = landmarks[LEFT_LIP_CORNER]
                right_lip = landmarks[RIGHT_LIP_CORNER]
                upper_lip = landmarks[UPPER_LIP_CENTER]
                lower_lip = landmarks[LOWER_LIP_CENTER]
                
                horizontal_distance = calculate_distance(left_lip, right_lip)
                vertical_distance = calculate_distance(upper_lip, lower_lip)
                
                if horizontal_distance > 0:
                    ratio = vertical_distance / horizontal_distance
                else:
                    ratio = 0
                
                total_ratio += ratio
                frame_count += 1
                
                # Draw lines and points on frame
                cv2.line(frame, tuple(left_lip.astype(int)), tuple(right_lip.astype(int)), (255, 0, 0), 2)
                cv2.line(frame, tuple(upper_lip.astype(int)), tuple(lower_lip.astype(int)), (0, 255, 0), 2)
                cv2.circle(frame, tuple(left_lip.astype(int)), 3, (0, 255, 255), -1)
                cv2.circle(frame, tuple(right_lip.astype(int)), 3, (0, 255, 255), -1)
                cv2.circle(frame, tuple(upper_lip.astype(int)), 3, (255, 0, 0), -1)
                cv2.circle(frame, tuple(lower_lip.astype(int)), 3, (255, 0, 0), -1)
                
                # Display ratio
                cv2.putText(frame, f"Lip Ratio: {ratio:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Lip Movement Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    avg_ratio = total_ratio / frame_count if frame_count > 0 else 0
    return avg_ratio

if __name__ == "__main__":
    video_path = "video.mp4"  # Replace with actual video file path
    avg_lip_ratio = lip_movement(video_path)
    print(f"Average Lip Movement Ratio: {avg_lip_ratio:.2f}")