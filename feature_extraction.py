import cv2
import pandas as pd
import os
from feature import ratina
from feature import blink
from feature import face
from feature import lipMovement
from feature import lipWrinkle

def extract_features(video_path, label):
    try:
        avg_ratio = ratina.ratina_distance(video_path)
        total_blinks, blink_rate = blink.count_blinks(video_path)
        face_ratio, face_fake = face.face_movement(video_path)
        avg_lip_ratio = lipMovement.lip_movement(video_path)
        wrinkle_ratio = lipWrinkle.wrinkle(video_path)
        
        data = {
            "Video Path": video_path,
            "Label": label,
            "Avg Retina Distance Ratio": avg_ratio,
            "Total Blinks": total_blinks,
            "Blink Rate (per min)": blink_rate,
            "Face Movement Ratio": face_ratio,
            "Face Fake Movement": "Yes" if face_fake else "No",
            "Avg Lip Movement Ratio": avg_lip_ratio,
            "Lip Wrinkle Ratio": wrinkle_ratio
        }
        return data
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def process_folder(folder_path, label, results):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video in video_files:
        video_path = os.path.join(folder_path, video)
        print(f"Processing: {video_path}")
        data = extract_features(video_path, label)
        if data:
            results.append(data)

if __name__ == "__main__":
    real_videos_folder = "real"  # Replace with actual folder path
    fake_videos_folder = "fake"  # Replace with actual folder path
    output_csv = os.path.join(os.getcwd(), "features.csv")
    
    results = []
    process_folder(real_videos_folder, "Real", results)
    process_folder(fake_videos_folder, "Fake", results)
    
    if results:
        df = pd.DataFrame(results)
        try:
            df.to_csv(output_csv, index=False)
            print(f"Feature extraction complete. Results saved to: {output_csv}")
        except PermissionError:
            print(f"Permission denied: Unable to write to {output_csv}. Close the file if open and try again.")
    else:
        print("No valid videos processed.")