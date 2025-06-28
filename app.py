import streamlit as st
import tempfile
import os
import base64
import google.generativeai as genai
from gtts import gTTS
import whisper
from pytube import YouTube
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(page_title="Video Insights", layout="wide")
st.title("🎬 Video Insights")

# --- Sidebar ---
st.sidebar.title("🔐 API Key")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# --- Model Setup ---
def init_model():
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")  # Using gemini-pro which is more widely available
        return model
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

# --- Utility: Download & Transcribe Video ---
def download_and_transcribe(video_url):
    try:
        # Validate URL
        if "youtube.com" not in video_url and "youtu.be" not in video_url:
            st.error("Please enter a valid YouTube URL")
            return None, None
        
        # Download video
        yt = YouTube(video_url)
        stream = yt.streams.filter(only_audio=True).first()
        
        if not stream:
            st.error("Could not find an audio stream for this video")
            return None, None
            
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        stream.download(filename=temp_audio.name)
        
        # Transcribe audio
        model = whisper.load_model("base")
        result = model.transcribe(temp_audio.name)
        
        # Clean up temp file
        os.remove(temp_audio.name)
        
        return result["text"], yt.title
    except Exception as e:
        logger.error(f"Error in download_and_transcribe: {str(e)}")
        st.error(f"Error processing video: {str(e)}")
        return None, None

# --- Utility: Gemini Content Generator ---
def gemini_generate(text, prompt, max_attempts=3):
    model = init_model()
    if not model:
        return "Please provide a valid API key."
    
    full_prompt = f"{prompt}\n\n{text[:8000]}"  # Limit text length to avoid token limits
    
    for attempt in range(max_attempts):
        try:
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt+1}/{max_attempts}): {str(e)}")
            if attempt == max_attempts - 1:
                return f"Error generating content: {str(e)}"
            # Wait before retrying
            import time
            time.sleep(2)

# --- Utility: TTS ---
def text_to_audio(text, filename="summary.mp3"):
    try:
        # Limit text length for TTS
        text_for_tts = text[:3000]  # Limit to prevent issues with very long texts
        tts = gTTS(text=text_for_tts, lang='en')
        tts.save(filename)
        return filename
    except Exception as e:
        logger.error(f"Error in text_to_audio: {str(e)}")
        st.error(f"Error generating audio: {str(e)}")
        return None

# --- Check API Key Validity ---
def check_api_key(api_key):
    if not api_key:
        return False
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Hello")
        return True
    except Exception as e:
        logger.error(f"API key validation error: {str(e)}")
        return False

# --- Main App UI ---
st.subheader("Step 1: Provide Video")
video_url = st.text_input("Paste YouTube video URL:")

# Process video when URL is provided
if video_url:
    # First check if API key is valid
    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar")
    elif not check_api_key(api_key):
        st.error("Invalid API key. Please check your Gemini API key and try again.")
    else:
        with st.spinner("Downloading and transcribing video..."):
            transcript, title = download_and_transcribe(video_url)
        
        if transcript and title:
            st.success("Video processed successfully!")
            st.write(f"**Video Title:** {title}")

            st.subheader("Step 2: AI-Powered Analysis")
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Summary", "Q&A", "Key Points", "Transcript", "Audio"
            ])

            with tab1:
                st.markdown("**Summary:**")
                summary = gemini_generate(transcript, "Summarize this video content in a concise and engaging way:")
                st.write(summary)

            with tab2:
                question = st.text_input("Ask a question about the video:")
                if question:
                    with st.spinner("Generating answer..."):
                        answer = gemini_generate(transcript, f"Answer this based on the video transcript: {question}")
                        st.write("**Answer:**", answer)

            with tab3:
                st.markdown("**Key Points:**")
                with st.spinner("Extracting key points..."):
                    key_points = gemini_generate(transcript, "List the 5-7 most important key points from the video:")
                    st.write(key_points)

            with tab4:
                st.markdown("**Transcript:**")
                st.text_area("Full transcript", transcript, height=300)
                st.download_button("Download Transcript", transcript, file_name="transcript.txt")

            with tab5:
                st.markdown("**Audio Summary:**")
                with st.spinner("Generating audio..."):
                    audio_file = text_to_audio(summary)
                    if audio_file:
                        st.audio(audio_file, format="audio/mp3")
                        
                        # Download button for audio
                        with open(audio_file, "rb") as f:
                            audio_bytes = f.read()
                        st.download_button(
                            label="Download Audio Summary",
                            data=audio_bytes,
                            file_name="summary.mp3",
                            mime="audio/mp3"
                        )
else:
    st.info("Enter a YouTube video link and your Gemini API key to begin.")

# --- Footer ---
st.markdown("---")
st.markdown("Video Insights - Analyze any YouTube video content with AI")
