import joblib # type: ignore
import pandas as pd # type: ignore
import sys
from feature_extraction import extract_features
import os

# Load trained model
# model = joblib.load("deepfake_model.pkl")
model = joblib.load("model_rf.pkl")

def predict_video(video_path):
    print(f"Processing video: {video_path}")

    # Extract features from the new video
    features = extract_features((video_path, None))  # Pass both video_path and label
    
    if features is None:
        print("Error extracting features. Check the video file.")
        return
    
    # Convert features to DataFrame
    feature_df = pd.DataFrame([features])
    feature_df = feature_df.drop(columns=["Video Path", "Label", "Total Blinks","Face Fake Movement"], errors='ignore')
    
    # Predict
    prediction = model.predict(feature_df)
    result = "Fake" if prediction[0] == 1 else "Real"
    print(f"Prediction: The video is {result}")
    
    return result
    # print(feature_df)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    
    result = predict_video(video_path)
