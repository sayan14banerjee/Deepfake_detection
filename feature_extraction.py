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

def load_processed_paths(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return set(df["Video Path"].tolist())
    return set()

if __name__ == "__main__":
    real_videos_folder = "real"
    fake_videos_folder = "fake"
    output_csv = os.path.join(os.getcwd(), "features.csv")
    batch_size = 20

    all_tasks = []
    all_tasks.extend(gather_video_paths(real_videos_folder, "Real"))
    all_tasks.extend(gather_video_paths(fake_videos_folder, "Fake"))

    already_done = load_processed_paths(output_csv)
    tasks_to_run = [task for task in all_tasks if task[0] not in already_done]

    print(f"[i] Total videos: {len(all_tasks)}")
    print(f"[i] Already processed: {len(already_done)}")
    print(f"[i] Remaining to process: {len(tasks_to_run)}")

    if tasks_to_run:
        for i in range(0, len(tasks_to_run), batch_size):
            batch = tasks_to_run[i:i+batch_size]
            with Pool(cpu_count()) as pool:
                batch_results = pool.map(extract_features, batch)

            valid_batch = [r for r in batch_results if r]

            if valid_batch:
                df_batch = pd.DataFrame(valid_batch)
                if os.path.exists(output_csv):
                    df_batch.to_csv(output_csv, mode='a', header=False, index=False)
                else:
                    df_batch.to_csv(output_csv, index=False)
            print(f"[✓] Saved batch {i//batch_size + 1}")
    else:
        print("[✓] All videos already processed.")
