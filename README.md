# ANFIS Hybrid Facial Emotion and Stress Analyzer

Streamlit demo application for facial emotion recognition and emotional stress level estimation using an ANFIS Hybrid model.

This project was built for the Soft Computing course as an academic prototype. It combines CNN, LBP, HOG, and ANFIS-based fuzzy reasoning to classify facial emotion and estimate a stress score from facial expression input.

## Team

| Name | Student ID |
| --- | --- |
| Siti Aisyah Nurdyanti (Isya) | 140810230015 |
| Clarisya Adeline (Ica) | 140810230017 |
| Nazwa Nashatasya (Awa) | 140810230019 |

## Dataset

FER2013 from Kaggle:

- 35,887 original grayscale face images
- 48 x 48 pixels
- 7 emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

Training notebook reference:

- `(NEWEST)ANFIS_Hybrid_FacialEmotion_Final (1).ipynb`
- Full FER2013 train and test folders were loaded during training.
- Disgust class was augmented to reduce imbalance.
- Total training experiment data after augmentation: 38,840 images.

## Current App Features

| Feature | Description |
| --- | --- |
| Upload Image | Upload JPG, JPEG, or PNG image and run full analysis. |
| Webcam Capture | Capture a single frame using `st.camera_input` and run full analysis. |
| Real-Time Webcam | Run frame-by-frame emotion and stress inference using `streamlit-webrtc`. |
| Face Detection | OpenCV DNN ResNet SSD face detector with fallback to full image. |
| Mirror Correction | Capture and realtime modes correct camera mirroring so left/right orientation matches real-world direction. |
| Realtime Size Control | Slider to adjust realtime webcam display width from 35% to 100%. |
| Preprocessing | Grayscale conversion, face crop, resize to 48 x 48, CLAHE, and normalization. |
| Emotion Prediction | 7-class softmax output with confidence score. |
| Top 3 Predictions | Ranked probabilities for the three most likely emotions. |
| Stress Score | Continuous 0-100 stress score with Low, Moderate, and High category. |
| Probability Chart | Horizontal bar chart for all 7 emotion probabilities. |
| Stress Gauge | Semicircle gauge showing stress score and category scale. |
| LBP Map | Local Binary Pattern visualization for facial texture. |
| HOG Map | Histogram of Oriented Gradients visualization for facial structure. |
| LBP Histogram | First 64 bins from the 256-bin LBP feature vector. |
| Feature Dimension Table | Summary of CNN, LBP, HOG, and fused feature dimensions. |
| Architecture Expander | Detailed model pipeline from input image to output heads. |
| Professional UI | Dashboard-style layout, project console sidebar, result cards, and equal-height panels. |
| Simulation Mode | Deterministic fallback when ANFIS weights are not available. |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure required face detector files exist

The app requires these OpenCV DNN face detector files in the same folder as `app.py`:

```text
deploy.prototxt
res10_300x300_ssd_iter_140000.caffemodel
```

These files are required. The app will stop if they are missing.

Official OpenCV sources:

- `deploy.prototxt`: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
- `res10_300x300_ssd_iter_140000.caffemodel`: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

### 3. Ensure ANFIS weights exist

The model weights file should be placed in the same folder as `app.py`:

```text
anfis_emotion_model.weights.h5
```

If this file exists, the app loads the trained ANFIS Hybrid weights.

If this file is missing, the app still runs in simulation mode. In simulation mode, preprocessing, feature extraction, UI, charts, and demo flow remain active, but predictions are generated deterministically from feature-based fallback logic.

### 4. Start the Streamlit app

```bash
streamlit run app.py
```

Default URL:

```text
http://localhost:8501
```

If port `8501` is already in use, Streamlit may open on another port such as `8502`.

## Input Modes

| Mode | Trigger | Output |
| --- | --- | --- |
| Upload Image | User uploads image file | Full dashboard with emotion card, stress card, charts, LBP/HOG maps, histogram, dimensions, and architecture. |
| Webcam Capture | User captures one frame | Same full dashboard as upload mode. |
| Real-Time Webcam | Continuous webcam stream | Live overlay with face bounding box, emotion confidence, and stress score. Heavy charts are skipped for performance. |

## Camera Orientation

The app includes mirror correction flags:

```python
REALTIME_MIRROR_CORRECTION = True
CAPTURE_MIRROR_CORRECTION = True
```

These settings make webcam orientation match real-world direction. For example, if a hand is on the user's left side in real life, it should also appear on the left side in the app.

If a specific device or browser already provides non-mirrored frames, these flags can be changed to `False`.

## Real-Time Webcam Details

Real-time mode uses `streamlit-webrtc` and the `RealtimeEmotionProcessor` class.

Optimizations:

| Optimization | Detail |
| --- | --- |
| Frame skipping | Inference runs every 5 frames. |
| Processing resolution | Frames are resized to 640 x 480 before processing. |
| Face detector reuse | `FACE_NET` is initialized once globally. |
| Lightweight output | Heavy Matplotlib charts are not rendered in realtime mode. |
| FPS limit | Webcam constraint uses `frameRate: {ideal: 15, max: 15}`. |
| Safe handling | Exceptions are caught to prevent stream crashes. |
| Display size slider | User can adjust realtime display width from 35% to 100%. |

Prediction smoothing:

```text
Sliding window: 8 recent predictions
Exponential smoothing alpha: 0.60
smoothed = 0.60 * current_mean + 0.40 * previous_smoothed
```

Realtime overlay:

```text
Emotion: Happy (87.3%)
Stress: 18/100 - LOW
Bounding box color:
- Green: Low
- Orange: Moderate
- Red: High
```

## Model Architecture

```text
Input Image (48 x 48 x 1)
      |
      +-- CNN Branch
      |      Resize to 96 x 96 x 3
      |      MobileNetV2
      |      Dense(256) + BatchNorm + Dropout
      |      Output: 256-dim
      |
      +-- LBP Branch
      |      Multi-radius LBP feature input
      |      BatchNorm + Dense(128, tanh) + Dropout
      |      Output: 128-dim projection
      |
      +-- HOG Branch
             HOG feature input
             BatchNorm + Dense(128, tanh) + Dropout
             Output: 128-dim projection

Feature Fusion:
      256 + 128 + 128 = 512-dim
      Cross-attention fusion
      Dense(256) + BatchNorm + Dropout
      Dense(64, tanh) + LayerNorm

ANFIS Core:
      L1: Dual fuzzification
          Gaussian MF + Generalized Bell MF
      L2: Fuzzy rule layer
          48 rules, T-norm product
      L3: Normalization layer
          w_bar_k = w_k / sum(w_j)
      L4: TSK consequent layer
      L5: Defuzzification
          LayerNorm + GELU + Dropout

Residual Connection:
      ANFIS output + projected fused features

Output Heads:
      Emotion Head: Dense(64) -> Dense(32) -> Dense(7, softmax)
      Stress Head : Dense(64) -> Dense(32) -> Dense(1, sigmoid) x 100
```

ANFIS configuration:

| Parameter | Value |
| --- | --- |
| Fuzzy membership functions | 5 |
| Fuzzy rules | 48 |
| ANFIS dimension | 128 |
| Compress dimension | 64 |
| CNN features | 256 |
| LBP features | 256 raw, 128 projected |
| HOG features | 324 raw, 128 projected |
| Fused projected dimension | 512 |

## Preprocessing Pipeline

The same preprocessing pipeline is used for upload, capture, and realtime modes:

```text
1. Convert image to grayscale.
2. Detect face using OpenCV DNN ResNet SSD.
3. Select largest detected face.
4. Convert face box to square crop.
5. Fallback to full image if no face is detected.
6. Resize to 48 x 48 pixels.
7. Apply CLAHE with clipLimit=2.0 and tileGridSize=4 x 4.
8. Normalize pixel values to [0, 1].
```

Feature extraction:

```text
LBP:
  Multi-radius uniform LBP
  R=3, P=24, 128 bins
  R=5, P=16, 128 bins
  Concatenated output: 256 bins
  L1-normalized

HOG:
  orientations=12
  pixels_per_cell=6 x 6
  cells_per_block=2 x 2
  Output: 324-dim
  L2-normalized
```

## Stress Score

The app has two stress score paths:

### 1. Loaded model mode

When `anfis_emotion_model.weights.h5` is available, stress is produced by the trained model:

```text
stress_score = sigmoid_output x 100
```

### 2. Simulation fallback mode

When weights are missing, stress is generated from emotion-to-stress weights plus small noise:

```text
stress_score = emotion_weight x 100 + noise
```

The score is clipped to the range 0-100.

Stress categories:

| Range | Category |
| --- | --- |
| 0-33 | Low |
| 34-66 | Moderate |
| 67-100 | High |

Emotion-to-stress weights:

| Emotion | Weight | Level |
| --- | ---: | --- |
| Fear | 0.90 | High |
| Angry | 0.85 | High |
| Sad | 0.75 | High |
| Disgust | 0.70 | High by app threshold |
| Surprise | 0.45 | Moderate |
| Neutral | 0.20 | Low |
| Happy | 0.05 | Low |

## Project Structure

```text
facial-expression-distress-analysis/
|-- app.py
|-- README.md
|-- requirements.txt
|-- anfis_emotion_model.weights.h5
|-- deploy.prototxt
|-- res10_300x300_ssd_iter_140000.caffemodel
|-- (NEWEST)ANFIS_Hybrid_FacialEmotion_Final (1).ipynb
```

File roles:

| File | Description |
| --- | --- |
| `app.py` | Main Streamlit application. |
| `README.md` | Project documentation. |
| `requirements.txt` | Python dependencies. |
| `anfis_emotion_model.weights.h5` | Trained ANFIS Hybrid model weights. |
| `deploy.prototxt` | OpenCV DNN face detector architecture. |
| `res10_300x300_ssd_iter_140000.caffemodel` | OpenCV DNN face detector weights. |
| `(NEWEST)ANFIS_Hybrid_FacialEmotion_Final (1).ipynb` | Training and experiment notebook. |

## Dependencies

Main dependencies from `requirements.txt`:

```text
streamlit
streamlit-webrtc
opencv-python-headless
numpy
scikit-image
matplotlib
tensorflow
Pillow
pandas
scipy
av
```

## Important Notes

- This app is an academic prototype.
- Stress output is an estimated emotional distress indication derived from facial expression analysis and stress mapping logic.
- The system is not clinically validated.
- Do not use the output for medical, psychological, hiring, disciplinary, or diagnostic decisions.
- Real-world behavior may vary because of lighting, camera quality, head pose, occlusion, glasses, and FER2013's low-resolution grayscale training data.

## Disclaimer

This system is intended for academic and research demonstration only. It is not a clinical diagnosis tool and must not replace assessment by qualified medical, psychological, or mental-health professionals.
