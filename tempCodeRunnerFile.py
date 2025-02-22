import cv2
import pandas as pd
import ratina
import blink
import face
import lipMovement
import lipWrinkle

def extract_features(video_path, output_csv):
    # Run all feature extraction functions
    avg_ratio = ratina.ratina_distance(video_path)
    total_blinks, blink_rate = blink.count_blinks(video_path)
    face_ratio, face_fake = face.face_movement(video_path)
    avg_lip_ratio = lipMovement.lip_movement(video_path)
    wrinkle_ratio = lipWrinkle.wrinkle(video_path)
    
    # Store results in a dictionary
    data = {
        "Video Path": video_path,
        "Avg Retina Distance Ratio": avg_ratio,
        "Total Blinks": total_blinks,
        "Blink Rate (per min)": blink_rate,
        "Face Movement Ratio": face_ratio,
        "Face Fake Movement": "Yes" if face_fake else "No",
        "Avg Lip Movement Ratio": avg_lip_ratio,
        "Lip Wrinkle Ratio": wrinkle_ratio
    }
    
    # Convert dictionary to DataFrame and save to CSV
    df = pd.DataFrame([data])
    df.to_csv(output_csv, index=False)
    
    print("Feature extraction complete. Results saved to:", output_csv)
    return data

if __name__ == "__main__":
    video_path = "video.mp4"  # Replace with actual video file
    output_csv = "features.csv"
    extract_features(video_path, output_csv)
