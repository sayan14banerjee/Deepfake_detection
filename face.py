import cv2
import mediapipe as mp
import numpy as np

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

mp_face_mesh = mp.solutions.face_mesh # type: ignore 

LEFT_SIDE = 234  # Approximate point on the left side of the face
RIGHT_SIDE = 454  # Approximate point on the right side of the face

def face_movement(video_path):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    left_movement = 0
    right_movement = 0
    prev_distance = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark])
                
                left_side = landmarks[LEFT_SIDE]
                right_side = landmarks[RIGHT_SIDE]
                
                distance = calculate_distance(left_side, right_side)
                if prev_distance is not None:
                    if distance > prev_distance:
                        right_movement += (distance - prev_distance)
                    else:
                        left_movement += (prev_distance - distance)
                
                prev_distance = distance
                
                # Draw lines and points on frame
                cv2.line(frame, tuple(left_side.astype(int)), tuple(right_side.astype(int)), (255, 0, 0), 2)
                cv2.circle(frame, tuple(left_side.astype(int)), 3, (0, 255, 0), -1)
                cv2.circle(frame, tuple(right_side.astype(int)), 3, (0, 0, 255), -1)
                
                # Display distance and movement
                cv2.putText(frame, f"Distance: {distance:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"L-Move: {left_movement:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                cv2.putText(frame, f"R-Move: {right_movement:.2f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow("Head Movement Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    ratio = right_movement / left_movement if left_movement > 0 else float('inf')
    is_fake = ratio > 1.5 or ratio < 0.67  # If movement is not balanced
    
    return ratio, is_fake

if __name__ == "__main__":
    video_path = "video.mp4"  # Replace with actual video file path
    ratio, is_fake = face_movement(video_path)
    print(f"Movement Ratio (R/L): {ratio:.2f}")
    print("Fake Movement Detected" if is_fake else "Natural Movement")
