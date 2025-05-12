import cv2
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from feature import ratina
from feature import blink
from feature import face
from feature import lipMovement
from feature import lipWrinkle


def extract_features(args):
    video_path, label = args
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
        print(f"[✔] Processed: {os.path.basename(video_path)}")
        return data
    except Exception as e:
        print(f"[✘] Error processing {video_path}: {e}")
        return None


def gather_video_paths(folder_path, label):
    if not os.path.exists(folder_path):
        print(f"[!] Folder not found: {folder_path}")
        return []

    video_files = [
        os.path.abspath(os.path.join(folder_path, f))
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ]
    return [(video, label) for video in video_files]


if __name__ == "__main__":
    real_videos_folder = "real"
    fake_videos_folder = "fake"
    output_csv = os.path.join(os.getcwd(), "features.csv")

    # Collect tasks
    all_tasks = []
    all_tasks.extend(gather_video_paths(real_videos_folder, "Real"))
    all_tasks.extend(gather_video_paths(fake_videos_folder, "Fake"))

    print(f"[i] Total videos to process: {len(all_tasks)}")

    results = []
    if all_tasks:
        with Pool(cpu_count()) as pool:
            results = pool.map(extract_features, all_tasks)

    # Filter out None results
    valid_results = [r for r in results if r]

    if valid_results:
        df = pd.DataFrame(valid_results)
        try:
            df.to_csv(output_csv, index=False)
            print(f"\n[✓] Feature extraction complete. Results saved to: {output_csv}")
        except PermissionError:
            print(f"\n[✘] Permission denied: Unable to write to {output_csv}. Close the file if open and try again.")
    else:
        print("\n[!] No valid videos processed.")
