import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh  # type: ignore
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


# Define facial landmark points for lips and cheeks (closer to the nose)
LEFT_UPPER_LIP = 61
RIGHT_UPPER_LIP = 291
LEFT_LOWER_LIP = 146
RIGHT_LOWER_LIP = 375
LEFT_UPPER_CHEEK = 207
RIGHT_UPPER_CHEEK = 427
LEFT_LOWER_CHEEK = 169
RIGHT_LOWER_CHEEK = 394
A1 = 214
A2 = 434

# Function to calculate distance between two points
def calculate_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# Function to process video or webcam feed to calculate distances
def wrinkle(video_path):
    # Use video file if provided, otherwise use webcam
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to RGB (MediaPipe uses RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect facial landmarks
        result = face_mesh.process(rgb_frame)

        # If landmarks are detected
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Get points for lips and cheeks
                landmarks = [LEFT_UPPER_LIP, RIGHT_UPPER_LIP, LEFT_LOWER_LIP, RIGHT_LOWER_LIP,
                             LEFT_UPPER_CHEEK, RIGHT_UPPER_CHEEK, LEFT_LOWER_CHEEK, RIGHT_LOWER_CHEEK, A1, A2]

                points = {landmark: (int(face_landmarks.landmark[landmark].x * frame.shape[1]),
                                     int(face_landmarks.landmark[landmark].y * frame.shape[0]))
                          for landmark in landmarks}

                # Draw circles on the face
                for i, landmark in enumerate(landmarks):
                    if i < 4:  # Lips
                        color = (0, 255, 0)  # Green
                    else:  # Cheeks
                        color = (255, 0, 0)  # Blue
                    cv2.circle(frame, points[landmark], 5, color, -1)

                # Draw lines between specific landmarks
                cv2.line(frame, points[LEFT_UPPER_LIP], points[LEFT_UPPER_CHEEK], (0, 255, 255), 2)  # Yellow
                cv2.line(frame, points[RIGHT_UPPER_LIP], points[RIGHT_UPPER_CHEEK], (0, 255, 255), 2)
                cv2.line(frame, points[LEFT_LOWER_LIP], points[LEFT_LOWER_CHEEK], (0, 255, 255), 2)
                cv2.line(frame, points[RIGHT_LOWER_LIP], points[RIGHT_LOWER_CHEEK], (0, 255, 255), 2)

                cv2.line(frame, points[LEFT_UPPER_CHEEK], points[RIGHT_UPPER_CHEEK], (0, 0, 255), 1)  # Red
                cv2.line(frame, points[LEFT_UPPER_CHEEK], points[A1], (0, 0, 255), 1)
                cv2.line(frame, points[A2], points[RIGHT_UPPER_CHEEK], (0, 0, 255), 1)
                cv2.line(frame, points[LEFT_LOWER_CHEEK], points[RIGHT_LOWER_CHEEK], (0, 0, 255), 1)
                cv2.line(frame, points[LEFT_LOWER_CHEEK], points[A1], (0, 0, 255), 1)
                cv2.line(frame, points[A2], points[RIGHT_LOWER_CHEEK], (0, 0, 255), 1)
                cv2.line(frame, points[A1], points[A2], (0, 0, 255), 1)

                # Calculate distances
                dist_left_upper = calculate_distance(points[LEFT_UPPER_LIP], points[LEFT_UPPER_CHEEK])
                dist_right_upper = calculate_distance(points[RIGHT_UPPER_LIP], points[RIGHT_UPPER_CHEEK])

                # Display distances
                cv2.putText(frame, f"Left Lip to Cheek Dist: {dist_left_upper:.2f} px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Right Lip to Cheek Dist: {dist_right_upper:.2f} px", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow('Lip to Cheek Distances (Closer to Nose)', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    ratio = (dist_left_upper / dist_right_upper) if dist_right_upper > 0 else 0 # type: ignore
    return ratio

if __name__ == "__main__":
    video_path = "video.mp4"  # Replace with actual video file path
    ratio = wrinkle(video_path)
    print(f"Movement Ratio (R/L): {ratio:.2f}")