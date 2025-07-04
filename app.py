import streamlit as st
import tempfile
import os
import google.generativeai as genai
from gtts import gTTS
import whisper
import yt_dlp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Config ---
@st.cache_resource
def load_whisper_model():
    """Loads the Whisper model and caches it across reruns."""
    logger.info("Loading Whisper model...")
    model = whisper.load_model("base")
    logger.info("Whisper model loaded.")
    return model


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
        model = genai.GenerativeModel("gemini-1.5-flash")  # Using a more recent and capable model
        return model
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

# --- Utility: Download & Transcribe Video ---
@st.cache_data(show_spinner=False) # Cache results to avoid re-processing the same URL
def download_and_transcribe(video_url):
    """
    Downloads audio from a YouTube URL using yt-dlp, transcribes it, and returns the text and title.
    """
    temp_audio_path = None
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_audio_path = tmp.name

        # yt-dlp options to download the best audio and convert it to mp3
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': temp_audio_path.replace('.mp3', ''), # yt-dlp adds the extension
            'quiet': True,
            'noprogress': True,
            'noplaylist': True, # Ensure we only get one video
        }

        # Download the audio and extract info
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            video_title = info_dict.get('title', 'Untitled Video')

        # Transcribe the downloaded audio file
        model = load_whisper_model() # Use cached model
        result = model.transcribe(temp_audio_path)

        return result["text"], video_title

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error: {str(e)}")
        # Provide user-friendly error messages based on yt-dlp's output
        error_str = str(e).lower()
        if 'video unavailable' in error_str:
            st.error("The video is unavailable. It may be private or deleted.")
        elif 'age restricted' in error_str:
            st.error("This video is age-restricted and cannot be processed without authentication.")
        elif 'private video' in error_str:
            st.error("This is a private video and cannot be accessed.")
        elif 'not available in your country' in error_str:
            st.error("This video is region-blocked and not available in your location.")
        else:
            st.error(f"Failed to download video. It may not exist or is restricted. Please try another URL.")
        return None, None
    except Exception as e:
        logger.error(f"Error in download_and_transcribe: {str(e)}")
        st.error(f"An unexpected error occurred during processing: {str(e)}")
        return None, None
    finally:
        # Ensure the temporary file is cleaned up
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

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
@st.cache_data # Cache the validation result for the given key
def check_api_key(api_key):
    if not api_key:
        return False
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
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
    elif not check_api_key(api_key): # This will now use the cached result
        st.error("Invalid API key. Please check your Gemini API key and try again.")
    else:
        with st.spinner("Downloading and transcribing video... (This may take a moment for new videos)"):
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
