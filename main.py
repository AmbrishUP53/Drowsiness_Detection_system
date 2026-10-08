import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import streamlit.components.v1 as components
import drowsiness_logic

# 1. Page Configuration
st.set_page_config(
    page_title="GuardianAI | Real-Time Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark/Light mode adaptive CSS for clean cards
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3B82F6;
        margin-bottom: 0px;
    }
    .sub-text {
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    .custom-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    .status-badge-active {
        background-color: #DCFCE7; 
        color: #16A34A;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-drowsy {
        background-color: #FEE2E2; 
        color: #DC2626;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session States safely
if "is_drowsy" not in st.session_state:
    st.session_state.is_drowsy = False
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0

# Title & Subtitle
st.markdown('<p class="main-header">🚗 GuardianAI: Real-Time Driver Drowsiness Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Advanced AI-powered active safety system using MediaPipe Face Mesh, Deep Learning CNN, and Browser Web Audio.</p>', unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("### ⚙️ Control Panel")
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📖 Quick Instructions")
st.sidebar.info(
    "1. Click **START** on the video feed.\n"
    "2. Allow browser camera access.\n"
    "3. Keep your face front-facing & well-lit.\n"
    "4. Drowsiness triggers live browser alarm."
)
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🧠 Tech Stack")
st.sidebar.markdown("- **Face Mesh:** MediaPipe")
st.sidebar.markdown("- **Model:** Custom CNN")
st.sidebar.markdown("- **Preprocessing:** CLAHE Filter")
st.sidebar.markdown("- **Audio:** Web Audio API")

# ----------------- VIDEO PROCESSOR -----------------
class DrowsinessProcessor(VideoTransformerBase):
    def __init__(self):
        self.model, self.face_mesh = drowsiness_logic.load_detection_resources()
        self.closed_frame_count = 0
        self.total_alerts = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        processed_img, status, self.closed_frame_count, self.total_alerts = drowsiness_logic.process_frame(
            img, self.model, self.face_mesh, self.closed_frame_count, self.total_alerts
        )
        
        # Update session states directly from video thread
        if "ALERT" in status:
            st.session_state.is_drowsy = True
            st.session_state.alert_count = self.total_alerts
        else:
            st.session_state.is_drowsy = False
            
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# ----------------- MAIN LAYOUT -----------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### 📹 Live Camera Feed")
    webrtc_streamer(
        key="drowsiness-detector",
        video_transformer_factory=DrowsinessProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]} ]},
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.markdown("### 📊 Live Telemetry Dashboard")
    
    # Dynamic placeholder container to refresh telemetry stats instantly
    telemetry_placeholder = st.empty()

    with telemetry_placeholder.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### Driver State Monitor")
        
        if st.session_state.is_drowsy:
            st.markdown('<span class="status-badge-drowsy">🚨 DROWSY / ASLEEP</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge-active">🟢 ACTIVE / AWAKE</span>', unsafe_allow_html=True)
            
        st.markdown("---")
        st.metric(label="Total Alerts Triggered", value=st.session_state.alert_count)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Safety Notice")
        st.write("Ensure your face is adequately lit. The CLAHE filter enhances daylight/indoor visibility automatically.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- BROWSER AUDIO ALARM JUGAD ---
if st.session_state.is_drowsy:
    alarm_js = """
    <script>
    function playAlarm() {
        if (!window.audioCtx) {
            window.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (window.audioCtx.state === 'suspended') {
            window.audioCtx.resume();
        }
        var osc = window.audioCtx.createOscillator();
        var gain = window.audioCtx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.value = 1000;
        osc.connect(gain);
        gain.connect(window.audioCtx.destination);
        osc.start();
        setTimeout(function() { osc.stop(); }, 400);
    }
    if (!window.alarmInterval) {
        window.alarmInterval = setInterval(playAlarm, 500);
    }
    </script>
    """
    components.html(alarm_js, height=0, width=0)
else:
    stop_js = """
    <script>
    if (window.alarmInterval) {
        clearInterval(window.alarmInterval);
        window.alarmInterval = null;
    }
    </script>
    """
    components.html(stop_js, height=0, width=0)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #9CA3AF; font-size: 0.9rem;'>GuardianAI Drowsiness Detection System &bull; Built with Streamlit & WebRTC</p>", unsafe_allow_html=True)