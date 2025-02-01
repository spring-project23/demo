 import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Constants
FRAME_THRESHOLD = 3  # Number of consecutive frames before triggering an alert

# Streamlit UI
st.title("Single Human Detection Alarm in Bus")

uploaded_file = st.file_uploader("Upload CCTV Video", type=["mp4", "avi", "mov"])
if uploaded_file:
    # Save the uploaded video
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("uploaded_video.mp4")
    frame_count = 0
    single_human_frames = 0  # Counter for frames with only one human detected

    stframe = st.empty()  # Placeholder for video display

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)  # Run YOLO detection

        detected_humans = 0

        # Process detections
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class ID

                if cls == 0:  # Class 0 is 'person'
                    detected_humans += 1
                    color = (0, 255, 0)  # Green for humans

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Check if exactly one human is detected
        if detected_humans == 1:
            single_human_frames += 1
        else:
            single_human_frames = 0  # Reset if multiple or no humans are detected

        # Trigger Alert if a single human is detected for too long
        if single_human_frames > FRAME_THRESHOLD:
            st.warning("🚨 ALERT: A single human is left alone in the bus!")

        # Display processed frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()
    st.success("Video processing complete.")
