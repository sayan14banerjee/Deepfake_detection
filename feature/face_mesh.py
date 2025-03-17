import cv2
import mediapipe as mp
import time
import numpy as np

mp_face_mesh = mp.solutions.face_mesh # type: ignore

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        cv2.imshow("Face Mesh Points", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video.mp4"  # Replace with actual video file path
    process_video(video_path)