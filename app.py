import time
import cv2
import streamlit as st
from drowsiness_logic import load_detection_resources, process_frame
import winsound

# Page Configuration
st.set_page_config(page_title="Driver Drowsiness Detection System", layout="centered")

st.title("🚨 Driver Drowsiness Detection System")
st.write("Real-time monitoring system using MediaPipe Face Mesh & CNN model.")

# Sidebar for Metrics & Controls
st.sidebar.title("📊 Session Analytics")
metric_status = st.sidebar.empty()
metric_closed_frames = st.sidebar.empty()
metric_total_alerts = st.sidebar.empty()
metric_session_time = st.sidebar.empty()

# run_button = st.sidebar.checkbox("Start Monitoring", value=False)

# Checkbox ki jagah Buttons
col1, col2 = st.sidebar.columns(2)
start_clicked = col1.button("Start", type="primary")
stop_clicked = col2.button("Stop")

if "running" not in st.session_state:
    st.session_state.running = False

if start_clicked:
    st.session_state.running = True
if stop_clicked:
    st.session_state.running = False
    try:
        winsound.PlaySound(None, winsound.SND_ASYNC)
    except Exception:
        pass

@st.cache_resource
def cached_resources():
    return load_detection_resources()

with st.spinner("Loading AI model and face mesh... Please wait."):
    model, face_mesh = cached_resources()

stframe = st.empty()

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    closed_frame_count = 0
    total_alerts = 0
    start_time = time.time()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        frame = cv2.flip(frame, 1)
        
        # Process frame using modular logic file
        frame, status, closed_frame_count, total_alerts = process_frame(
            frame, model, face_mesh, closed_frame_count, total_alerts
        )

        # Update Sidebar Analytics Live
        session_duration = int(time.time() - start_time)
        metric_status.metric("System Status", status)
        metric_closed_frames.metric("Closed Frames Counter", closed_frame_count)
        metric_total_alerts.metric("Total Drowsy Alerts", total_alerts)
        metric_session_time.metric("Session Time (sec)", session_duration)

        # Display frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
else:
    st.info("👈 Check 'Start Monitoring' in the sidebar to launch the live system.")