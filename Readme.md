# 🪖 Helmet Detection System using YOLO11

An AI-powered helmet detection application built using **YOLO11, OpenCV, and Streamlit** to identify two-wheelers, helmet usage, and riders without helmets from uploaded images and videos.

The application provides visual detection results, confidence scores, class-wise analytics, and downloadable annotated media.

---

## 🚀 Project Overview

Road safety is an important concern, particularly for two-wheeler riders. Wearing a helmet helps reduce the risk of serious head injuries.

This project uses computer vision and deep learning to automate helmet detection from images and videos.

A custom-trained YOLO11 model identifies three object classes:

- **Bike** – Detects two-wheelers.
- **With Helmet** – Detects helmet-wearing riders.
- **Without Helmet** – Detects riders without helmets.

The trained model is integrated into a Streamlit web application for interactive detection and visualization.

## ✨ Features

### 1. Image Detection
- Upload images in JPG, JPEG, and PNG formats.
- Detect bikes, riders with helmets, and riders without helmets.
- Display bounding boxes and confidence scores.
- Generate class-wise detection summaries.
- Download annotated images.

### 2. Video Detection
- Upload MP4, AVI, MOV, and MKV videos.
- Perform frame-by-frame object detection.
- Generate annotated output videos.
- Display processed videos directly in the browser.
- Convert processed videos to H.264 for browser compatibility.
- Download processed videos.

### 3. Detection Analytics
- Class-wise detection counts.
- Confidence score statistics.
- Detection distribution charts.
- Adjustable confidence threshold.
- Configurable inference image size.
- Class visibility filtering.

## 🧠 Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Object Detection | YOLO11 |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Web Framework | Streamlit |
| Data Processing | NumPy, Pandas |
| Image Processing | Pillow |
| Video Conversion | FFmpeg / imageio-ffmpeg |
| Model Training | Google Colab |
| Dataset Platform | Roboflow |

## 📊 Model Information

| Parameter | Details |
|---|---|
| Model | YOLO11n |
| Task | Object Detection |
| Number of Classes | 3 |
| Input Image Size | 640 × 640 |
| Training Epochs | 100 |
| Training Platform | Google Colab |
| Model Checkpoint | best.pt |

### Class Mapping

| Class ID | Class Name |
|---|---|
| 0 | Bike |
| 1 | With Helmet |
| 2 | Without Helmet |

**Note:** The class order must remain consistent with the trained model's dataset configuration.

## 📈 Model Evaluation Results

The trained model was evaluated on the held-out test dataset.

| Evaluation Metric | Result |
|---|---:|
| Precision | 93.42% |
| Recall | 85.65% |
| mAP@50 | 92.42% |
| mAP@50-95 | 61.12% |

### Class-wise Performance

| Class | Precision | Recall | mAP@50 |
|---|---:|---:|---:|
| Bike | 95.20% | 91.10% | 94.90% |
| With Helmet | 92.00% | 87.20% | 93.40% |
| Without Helmet | 93.00% | 78.70% | 89.00% |

**Observation:** The Without Helmet class has comparatively lower recall. This indicates that some actual no-helmet instances may be missed. Further dataset improvements and model evaluation can help address this limitation.

## 📁 Project Structure

```text
Helmet_Detection/
│
├── app.py
│
├── best.pt
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
└── .venv/
```

| File | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `best.pt` | Trained YOLO11 model checkpoint |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `.gitignore` | Excludes unnecessary files |
| `.venv/` | Local Python virtual environment |

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project directory:

```bash
cd Helmet_Detection
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

### Step 3: Install PyTorch

For a CPU-only Windows environment:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Add the Trained Model

Place your trained `best.pt` file in the project root directory, alongside `app.py`.

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## 📦 Requirements

The project's `requirements.txt` contains:

```text
ultralytics==8.4.215
opencv-python==4.14.0.94
numpy==2.4.6
pandas
streamlit
Pillow
imageio-ffmpeg
```

PyTorch is installed separately in the CPU-only setup described above.

## 🌐 Deployment

The application can be deployed using **Streamlit Community Cloud**.

General deployment steps:

1. Push your project source code to GitHub.
2. Ensure `app.py` and `requirements.txt` are available in the repository.
3. Make the trained `best.pt` checkpoint accessible to the deployed application.
4. Open Streamlit Community Cloud.
5. Connect your GitHub repository.
6. Select `app.py` as the application entry point.
7. Deploy the application.

**Model hosting note:** GitHub's standard file-size limit is 100 MB per file. If `best.pt` exceeds this limit, use Git LFS or external model hosting and configure the application to retrieve the model securely.

For deployment, test video processing with shorter videos first because cloud resources and processing time are limited.

## 📚 Dataset

The project uses a helmet detection dataset obtained through Roboflow.

**Dataset:** [Helmet Detection – Roboflow Universe](https://universe.roboflow.com/aju57/helmet-detection-2cu9l)

**Project version:** 1

**Dataset license:** CC BY 4.0

Dataset attribution should be retained in accordance with the license requirements.

## ⚠️ Limitations

- Detection accuracy depends on lighting, image quality, camera angle, and object visibility.
- Small or partially obscured helmets may be difficult to detect.
- The model may produce false positives and false negatives.
- Video detection counts detections across frames, not unique vehicles.
- Processed videos currently do not preserve the original audio.
- The application does not currently include live webcam detection.

This application is intended as a computer vision demonstration and should not be treated as a fully automated traffic enforcement system.

## 🔮 Future Enhancements

Potential improvements for future versions:

- Number plate detection.
- License plate recognition using OCR.
- Triple-riding detection.
- Traffic violation record management.
- Violation evidence storage.
- Real-time webcam detection.
- Database integration.
- Advanced analytics dashboard.

## 👨‍💻 Author

**Sai Sushanth Sailla**

- LinkedIn: [linkedin.com/in/sai-sushanth-sailla](https://linkedin.com/in/sai-sushanth-sailla)
- GitHub: [github.com/saisushanthsailla](https://github.com/saisushanthsailla)

---

⭐ If you find this project interesting, consider starring the repository!

**Built with Python, YOLO11, OpenCV, and Streamlit.**