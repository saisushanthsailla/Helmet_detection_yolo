
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import subprocess
import os
from pathlib import Path
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import imageio_ffmpeg


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Helmet AI | Detection Dashboard",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

CLASS_NAMES = {
    0: "Bike",
    1: "With Helmet",
    2: "Without Helmet"
}

CLASS_COLORS = {
    "Bike": "#3498DB",
    "With Helmet": "#2ECC71",
    "Without Helmet": "#E74C3C"
}


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 24px;
        border-radius: 14px;
        background: linear-gradient(120deg, #172554, #1D4ED8);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 30px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #DBEAFE;
        font-size: 15px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 15px;
        border-radius: 12px;
    }

    .section-heading {
        font-size: 21px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🪖 Helmet Detection AI</h1>
        <p>
        Intelligent two-wheeler helmet detection using
        YOLO11 Computer Vision.
        </p>
        <p>
        Detect bikes, helmet usage and without-helmet cases
        through uploaded images and videos.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):

    return YOLO(str(model_path))


if not MODEL_PATH.exists():

    st.error(
        "Model file not found. Please place best.pt "
        "in the same folder as app.py."
    )

    st.stop()


try:

    model = load_model(MODEL_PATH)

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Detection Controls")

st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Choose Detection Mode",
    [
        "Image Detection",
        "Video Detection"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Model Configuration")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05
)

image_size = st.sidebar.selectbox(
    "Inference Image Size",
    [320, 480, 640, 800],
    index=2
)

st.sidebar.markdown("---")

st.sidebar.subheader("Class Visibility")

visible_classes = st.sidebar.multiselect(
    "Select classes to display",
    options=[
        "Bike",
        "With Helmet",
        "Without Helmet"
    ],
    default=[
        "Bike",
        "With Helmet",
        "Without Helmet"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Model: YOLO11\n\n"
    "Architecture: Object Detection\n\n"
    "Classes: 3"
)


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def get_class_name(class_id, names):

    if isinstance(names, dict):

        return names.get(
            int(class_id),
            CLASS_NAMES.get(int(class_id), "Unknown")
        )

    return CLASS_NAMES.get(int(class_id), "Unknown")


def extract_detections(result):

    records = []

    if result.boxes is None:
        return records

    for box in result.boxes:

        class_id = int(box.cls.item())
        confidence_score = float(box.conf.item())

        class_name = get_class_name(
            class_id,
            result.names
        )

        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().numpy().tolist()
        )

        records.append({
            "Class": class_name,
            "Confidence": confidence_score,
            "X1": round(x1, 2),
            "Y1": round(y1, 2),
            "X2": round(x2, 2),
            "Y2": round(y2, 2)
        })

    return records


def get_summary(records):

    summary = {
        "Bike": 0,
        "With Helmet": 0,
        "Without Helmet": 0
    }

    for record in records:

        class_name = record["Class"]

        if class_name in summary:

            summary[class_name] += 1

    return summary


def filter_result(result, visible_classes):

    records = extract_detections(result)

    filtered_records = [
        record for record in records
        if record["Class"] in visible_classes
    ]

    return filtered_records


def annotate_result(result, visible_classes):

    annotated = result.orig_img.copy()

    records = filter_result(
        result,
        visible_classes
    )

    for record in records:

        x1 = int(record["X1"])
        y1 = int(record["Y1"])
        x2 = int(record["X2"])
        y2 = int(record["Y2"])

        class_name = record["Class"]
        confidence_score = record["Confidence"]

        color_hex = CLASS_COLORS.get(
            class_name,
            "#FFFFFF"
        )

        color = tuple(
            int(color_hex[i:i+2], 16)
            for i in (5, 3, 1)
        )

        label = (
            f"{class_name} "
            f"{confidence_score:.2f}"
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return annotated, records


def display_metrics(summary, total=None):

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🚲 Bikes",
        summary["Bike"]
    )

    col2.metric(
        "🪖 With Helmet",
        summary["With Helmet"]
    )

    col3.metric(
        "⚠️ Without Helmet",
        summary["Without Helmet"]
    )

    if total is not None:

        st.caption(
            f"Total detections: {total}"
        )


def display_analytics(records, summary):

    st.markdown(
        '<div class="section-heading">Detection Analytics</div>',
        unsafe_allow_html=True
    )

    total = len(records)

    if total == 0:

        st.info("No objects detected.")
        return

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Class Distribution**")

        chart_data = pd.DataFrame({
            "Class": list(summary.keys()),
            "Detections": list(summary.values())
        })

        chart_data = chart_data[
            chart_data["Detections"] > 0
        ]

        if not chart_data.empty:

            st.bar_chart(
                chart_data.set_index("Class")
            )

    with col2:

        st.write("**Confidence Statistics**")

        confidence_values = [
            record["Confidence"]
            for record in records
        ]

        st.metric(
            "Average Confidence",
            f"{np.mean(confidence_values):.2%}"
        )

        st.metric(
            "Highest Confidence",
            f"{np.max(confidence_values):.2%}"
        )

        st.metric(
            "Total Detections",
            total
        )


def image_download_button(image_array):

    image = Image.fromarray(image_array)

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95
    )

    st.download_button(
        "⬇️ Download Annotated Image",
        data=buffer.getvalue(),
        file_name="helmet_detection_result.jpg",
        mime="image/jpeg",
        use_container_width=True
    )


# ============================================================
# 7. IMAGE DETECTION
# ============================================================

if mode == "Image Detection":

    st.markdown(
        '<div class="section-heading">📷 Image Analysis</div>',
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "🚀 Run Image Detection",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Running YOLO11 inference..."):

                image_array = np.array(image)

                result = model.predict(
                    source=image_array,
                    conf=confidence,
                    imgsz=image_size,
                    verbose=False
                )[0]

                annotated, records = annotate_result(
                    result,
                    visible_classes
                )

                annotated_rgb = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                summary = get_summary(records)

            st.success(
                "Image detection completed!"
            )

            st.markdown("### Detection Result")

            st.image(
                annotated_rgb,
                use_container_width=True
            )

            st.markdown("### Detection Summary")

            display_metrics(
                summary,
                len(records)
            )

            display_analytics(
                records,
                summary
            )

            st.markdown("### Detection Details")

            if records:

                df = pd.DataFrame(records)

                df["Confidence"] = df[
                    "Confidence"
                ].map(lambda x: f"{x:.2%}")

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No objects matched the selected classes "
                    "and confidence threshold."
                )

            image_download_button(
                annotated_rgb
            )


# ============================================================
# 8. VIDEO DETECTION
# ============================================================

elif mode == "Video Detection":

    st.markdown(
        '<div class="section-heading">🎥 Video Analysis</div>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:

        video_bytes = uploaded_video.getvalue()

        st.subheader("Original Video")

        st.video(video_bytes)

        if st.button(
            "🚀 Process Video",
            type="primary",
            use_container_width=True
        ):

            input_path = None
            output_path = None
            h264_path = None

            try:

                # ------------------------------------------------
                # SAVE INPUT VIDEO
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_video.name).suffix
                ) as input_file:

                    input_file.write(video_bytes)
                    input_path = input_file.name

                # ------------------------------------------------
                # OPEN VIDEO
                # ------------------------------------------------

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():

                    raise RuntimeError(
                        "Unable to open uploaded video."
                    )

                fps = cap.get(cv2.CAP_PROP_FPS)

                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                if fps <= 0:
                    fps = 25.0

                if width <= 0 or height <= 0:

                    cap.release()

                    raise RuntimeError(
                        "Invalid video dimensions."
                    )

                # ------------------------------------------------
                # CREATE OUTPUT VIDEO
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as output_file:

                    output_path = output_file.name

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )

                if not writer.isOpened():

                    cap.release()

                    raise RuntimeError(
                        "Unable to create output video."
                    )

                # ------------------------------------------------
                # PROCESS VIDEO FRAMES
                # ------------------------------------------------

                progress_bar = st.progress(0)

                status_text = st.empty()

                frame_count = 0

                video_summary = {
                    "Bike": 0,
                    "With Helmet": 0,
                    "Without Helmet": 0
                }

                all_confidences = []

                st.info(
                    "Processing video frame by frame. "
                    "Please wait..."
                )

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    result = model.predict(
                        source=frame,
                        conf=confidence,
                        imgsz=image_size,
                        verbose=False
                    )[0]

                    annotated, records = annotate_result(
                        result,
                        visible_classes
                    )

                    writer.write(annotated)

                    frame_summary = get_summary(
                        records
                    )

                    for class_name in video_summary:

                        video_summary[class_name] += (
                            frame_summary[class_name]
                        )

                    for record in records:

                        all_confidences.append(
                            record["Confidence"]
                        )

                    frame_count += 1

                    if total_frames > 0:

                        progress = min(
                            frame_count / total_frames,
                            1.0
                        )

                        progress_bar.progress(
                            progress
                        )

                    status_text.text(
                        f"Processed {frame_count} / "
                        f"{total_frames if total_frames > 0 else '?'} frames"
                    )

                cap.release()
                writer.release()

                if frame_count == 0:

                    raise RuntimeError(
                        "No video frames could be processed."
                    )

                progress_bar.progress(1.0)

                status_text.success(
                    f"Successfully processed {frame_count} frames."
                )

                # ------------------------------------------------
                # CONVERT VIDEO TO H.264
                # ------------------------------------------------

                st.subheader(
                    "Preparing Browser-Compatible Video"
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as converted_file:

                    h264_path = converted_file.name

                ffmpeg_exe = (
                    imageio_ffmpeg.get_ffmpeg_exe()
                )

                command = [
                    ffmpeg_exe,
                    "-y",
                    "-i", output_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an",
                    h264_path
                ]

                conversion = subprocess.run(
                    command,
                    capture_output=True,
                    text=True
                )

                if conversion.returncode != 0:

                    raise RuntimeError(
                        "H.264 conversion failed:\n"
                        + conversion.stderr[-2500:]
                    )

                # ------------------------------------------------
                # READ CONVERTED VIDEO
                # ------------------------------------------------

                with open(h264_path, "rb") as file:

                    processed_video = file.read()

                if not processed_video:

                    raise RuntimeError(
                        "Converted video is empty."
                    )

                # ------------------------------------------------
                # DISPLAY PROCESSED VIDEO
                # ------------------------------------------------

                st.success(
                    "Video processing completed!"
                )

                st.subheader(
                    "Processed Video"
                )

                st.video(
                    processed_video,
                    format="video/mp4"
                )

                # ------------------------------------------------
                # VIDEO SUMMARY
                # ------------------------------------------------

                st.subheader(
                    "Video Detection Summary"
                )

                total_detections = sum(
                    video_summary.values()
                )

                display_metrics(
                    video_summary,
                    total_detections
                )

                # ------------------------------------------------
                # VIDEO ANALYTICS
                # ------------------------------------------------

                st.subheader(
                    "Video Analytics"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Frames Processed",
                    frame_count
                )

                col2.metric(
                    "Total Detections",
                    total_detections
                )

                if all_confidences:

                    col3.metric(
                        "Average Confidence",
                        f"{np.mean(all_confidences):.2%}"
                    )

                else:

                    col3.metric(
                        "Average Confidence",
                        "N/A"
                    )

                chart_data = pd.DataFrame({
                    "Class": list(video_summary.keys()),
                    "Detections": list(video_summary.values())
                })

                st.bar_chart(
                    chart_data.set_index("Class")
                )

                st.caption(
                    "Video detection counts are accumulated "
                    "across frames. The same vehicle may be "
                    "counted multiple times."
                )

                # ------------------------------------------------
                # DOWNLOAD PROCESSED VIDEO
                # ------------------------------------------------

                st.download_button(
                    "⬇️ Download Processed Video",
                    data=processed_video,
                    file_name="helmet_detection_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

            except Exception as error:

                st.error(
                    f"Video processing failed: {error}"
                )

            finally:

                # ------------------------------------------------
                # CLEAN UP TEMPORARY FILES
                # ------------------------------------------------

                for path in [
                    input_path,
                    output_path,
                    h264_path
                ]:

                    if path and os.path.exists(path):

                        try:

                            os.remove(path)

                        except OSError:

                            pass


# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:10px;">

    **Helmet Detection AI Dashboard**

    Powered by YOLO11 | OpenCV | Streamlit

    Image Detection • Video Detection

    </div>
    """,
    unsafe_allow_html=True
)
