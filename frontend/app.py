import streamlit as st
import requests
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime

# --- Logging Configuration ---
# Sets up logging to console for development/test visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MedicalScribeApp")

# --- Constants & Config ---
API_URL = os.getenv("SCRIBE_API_URL", "http://localhost:8000/scribe/process")
PAGE_CONFIG = {"page_title": "AI Medical Scribe", "page_icon": "🩺", "layout": "wide"}
CUSTOM_CSS = """
<style>
    .reportview-container { background: #f0f2f6; }
    .transcript-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #4a90e2;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .med-row { background-color: #ffffff; padding: 10px; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 10px; }
</style>
"""

# --- Service Layer ---
def process_audio_consultation(audio_file: Any) -> Optional[Dict[str, Any]]:
    """
    Sends audio file to backend API and processes the response.
    """
    logger.info("Initiating audio processing request.")
    try:
        files = {"file": ("consultation.wav", audio_file, "audio/wav")}
        
        # Log the attempt for dev tracking
        logger.debug(f"Posting to {API_URL}")
        
        response = requests.post(API_URL, files=files, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                logger.info("API request successful. Data received.")
                return result.get("data")
            else:
                logger.error(f"Backend processing failed: {result.get('message')}")
                st.error(f"Backend Error: {result.get('message')}")
        else:
            logger.error(f"HTTP Error {response.status_code}: {response.text}")
            st.error(f"Server returned status: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network/Request failed: {str(e)}")
        st.error(f"Connection Error: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred.")
        
    return None

# --- UI Components ---
def render_sidebar():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=50)
        st.title("Doctor's Tools")
        st.info(f"System Ready\nTime: {datetime.now().strftime('%H:%M')}")

def render_transcript_section():
    st.subheader("🎙️ Dictation")
    audio_value = st.audio_input("Record Prescription Instructions")

    if audio_value:
        logger.info("Audio input detected.")
        if st.button("🚀 Generate Prescription", type="primary"):
            with st.spinner("Processing consultation..."):
                data = process_audio_consultation(audio_value)
                if data:
                    st.session_state["result"] = data
                    logger.info("Session state updated with prescription data.")
                    st.success("Transcription Complete!")

    if "result" in st.session_state:
        st.markdown("### 📝 Full Transcript")
        raw_text = st.session_state["result"].get("raw_text", "No text available.")
        st.markdown(f'<div class="transcript-box">{raw_text}</div>', unsafe_allow_html=True)

def render_medicine_form():
    st.subheader("💊 Medicines (To be Prescribed)")
    
    if "result" not in st.session_state:
        st.info("Record and process audio to populate the medicine list.")
        return

    data = st.session_state["result"]
    medicines = data.get("medicines", [])
    
    logger.debug(f"Rendering form for {len(medicines)} medicines.")

    with st.form("prescription_form"):
        for i, med in enumerate(medicines):
            st.markdown(f"**Medicine #{i+1}**")
            
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.text_input("Name", value=med.get("name", ""), key=f"name_{i}")
            with c2:
                st.text_input("Dosage", value=med.get("dosage", ""), key=f"dose_{i}")
            with c3:
                st.text_input("Days", value=med.get("days", ""), key=f"days_{i}")
            
            c4, c5 = st.columns([2, 2])
            with c4:
                timing_options = ["After Food", "Before Food", "With Food"]
                current_timing = med.get("timing", "After Food")
                idx = timing_options.index(current_timing) if current_timing in timing_options else 0
                st.selectbox("Timing", timing_options, index=idx, key=f"time_{i}")
            with c5:
                st.text_input("Frequency", value=med.get("frequency", ""), key=f"freq_{i}")
            
            st.divider()
        
        if st.form_submit_button("✅ Finalize Prescription"):
            logger.info("Prescription finalized by user.")
            st.toast("Prescription Data Ready for Database!", icon="💾")
            st.balloons()
            # Here you would typically trigger a DB save function

# --- Main Application Flow ---
def main():
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    render_sidebar()
    
    st.title("🩺 AI Prescription Assistant")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        render_transcript_section()
        
    with col_right:
        render_medicine_form()

if __name__ == "__main__":
    main()