# Deepfake Detection System 🎭

A machine learning-based system to detect deepfake videos using various facial features like blink rate, lip movement, face movement, and retina tracking.

## 📁 Project Structure

```
deepfake_detection/
├── deepfake_model.pkl               # Trained ML model
├── features.csv                    # Dataset of extracted features
├── feature_extraction.py          # Extracts visual features from video
├── train.ipynb                    # Jupyter notebook to train the model
├── test.py                        # Script to test new video files
├── video.mp4 / video1.mp4         # Sample videos
└── feature/                       # Custom modules (ratina, blink, face, etc.)
```

## ⚙️ Features Extracted

- **Retina Distance Ratio**
- **Blink Rate and Count**
- **Face Movement Ratio**
- **Lip Movement and Wrinkle Ratio**

These features are extracted using OpenCV and custom heuristics, then fed into a machine learning model for classification.

## 🚀 How It Works

1. A video is passed into the system.
2. Facial features are extracted using OpenCV-based logic.
3. Features are saved in a CSV file (`features.csv`).
4. A model is trained using these features (see `train.ipynb`).
5. The model is saved as `deepfake_model.pkl`.
6. You can test new videos using `test.py`.

## 🖥️ Usage

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Extract Features from a New Video

```python
from feature_extraction import extract_features

video_path = 'video.mp4'
label = 0  # 0 for real, 1 for fake
features = extract_features(video_path, label)
```

### 3. Train Model

Open `train.ipynb` and run all cells to train the model using `features.csv`.

### 4. Predict Using Test Script

```bash
python test.py video.mp4
```

This will load `deepfake_model.pkl`, extract features from new videos, and make predictions.

## 📦 Requirements

- Python 3.x
- OpenCV
- Pandas
- Scikit-learn
- NumPy

## 📌 Notes

- Ensure that all feature scripts (`ratina`, `blink`, etc.) are working and properly imported.
- More robust detection can be achieved using deep learning (e.g., CNN, RNN) on video frames.

## 📜 License

MIT License – feel free to use and modify.

## 🤝 Contributing

Pull requests and feedback are welcome!
