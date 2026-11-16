
import streamlit as st
import av
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load trained YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("Live Helmet Detection")
st.write("Real-time helmet detection using YOLO11")

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

class VideoProcessor(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO prediction
        results = model.predict(
            source=img,
            conf=confidence,
            imgsz=640,
            verbose=False
        )

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        return annotated_frame


webrtc_streamer(
    key="helmet-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
