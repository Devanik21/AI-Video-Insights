import streamlit as st
import tempfile
import os
import base64
import google.generativeai as genai
from gtts import gTTS
from pytube import YouTube
import whisper
import json

# --- Page Config ---
st.set_page_config(page_title="Video Summarizer Pro", layout="wide")
st.title("🎬 Video Summarizer Pro")

# --- Sidebar ---
st.sidebar.title("🔐 API Key")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# --- Model Setup ---
def init_model():
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    else:
        return None

# --- Utility: Download & Transcribe Video ---
def download_and_transcribe(video_url):
    yt = YouTube(video_url)
    stream = yt.streams.filter(only_audio=True).first()
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    stream.download(filename=temp_audio.name)

    model = whisper.load_model("base")
    result = model.transcribe(temp_audio.name)
    os.remove(temp_audio.name)
    return result["text"], yt.title

# --- Utility: Gemini Content Generator ---
def gemini_generate(text, prompt):
    model = init_model()
    if not model:
        return "Please provide a valid API key."
    full_prompt = f"{prompt}\n\n{text}"
    response = model.generate_content(full_prompt)
    return response.text

# --- Utility: TTS ---
def text_to_audio(text, filename="summary.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

# --- Upload Section ---
st.subheader("Step 1: Provide Video")
video_url = st.text_input("Paste YouTube video URL:")

if video_url and api_key:
    with st.spinner("Downloading and transcribing video..."):
        transcript, title = download_and_transcribe(video_url)

    st.success("Video processed successfully!")
    st.write(f"**Video Title:** {title}")

    st.subheader("Step 2: AI-Powered Analysis")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Summary", "Q&A", "Flashcards", "Quiz", "Key Points",
        "Transcript", "Audio Overview", "Sentiment", "Topic Chart", "Export"
    ])

    with tab1:
        st.markdown("**Summary:**")
        summary = gemini_generate(transcript, "Summarize this video content in a concise and engaging way:")
        st.write(summary)

    with tab2:
        question = st.text_input("Ask a question about the video:")
        if question:
            answer = gemini_generate(transcript, f"Answer this based on the video: {question}")
            st.write("**Answer:**", answer)

    with tab3:
        st.markdown("**Flashcards:**")
        flashcards = gemini_generate(transcript, "Generate 5 flashcards with questions and answers:")
        st.code(flashcards, language="markdown")

    with tab4:
        st.markdown("**Quiz:**")
        quiz = gemini_generate(transcript, "Create a 5-question multiple choice quiz:")
        st.code(quiz, language="markdown")

    with tab5:
        st.markdown("**Key Points:**")
        key_points = gemini_generate(transcript, "List key points from the video:")
        st.write(key_points)

    with tab6:
        st.markdown("**Transcript with Timestamps:**")
        st.code(transcript, language="text")

    with tab7:
        st.markdown("**Voiceover Summary:**")
        audio_summary = text_to_audio(summary)
        st.audio(audio_summary, format="audio/mp3")

    with tab8:
        st.markdown("**Sentiment Analysis (coming soon...)**")
        st.info("This feature will analyze emotional tone throughout the video.")

    with tab9:
        st.markdown("**Topic Visualization (coming soon...)**")
        st.info("This will show topic shifts using visual graphs.")

    with tab10:
        st.markdown("**Export Options:**")
        st.download_button("Download Summary", summary, file_name="summary.txt")
        st.download_button("Download Transcript", transcript, file_name="transcript.txt")
        with open(audio_summary, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:audio/mp3;base64,{b64}" download="summary.mp3">Download Audio</a>'
            st.markdown(href, unsafe_allow_html=True)
else:
    st.info("Enter a YouTube video link and your Gemini API key to begin.")
