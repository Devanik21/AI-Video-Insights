import streamlit as st
import tempfile
import os
import base64
import google.generativeai as genai
from gtts import gTTS
from pytube import YouTube
from pytube.exceptions import PytubeError, RegexMatchError # Import specific exceptions
import whisper
import time # For potentially adding delays if needed

# --- Constants ---
TTS_FILENAME = "summary_audio.mp3"
# Use a known working model - check Google AI Studio for latest options
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

# --- Page Config ---
st.set_page_config(page_title="Video Summarizer Pro", layout="wide")
st.title("🎬 Video Summarizer Pro")
st.markdown("Summarize YouTube videos, ask questions, generate quizzes, and more!")
st.info("Ensure you have the latest `pytube` installed: `pip install --upgrade pytube`")

# --- Session State Initialization ---
# Initialize session state variables if they don't exist
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'video_url' not in st.session_state:
    st.session_state.video_url = ""
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'video_title' not in st.session_state:
    st.session_state.video_title = None
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'qa_answer' not in st.session_state:
    st.session_state.qa_answer = None
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = None
if 'quiz' not in st.session_state:
    st.session_state.quiz = None
if 'key_points' not in st.session_state:
    st.session_state.key_points = None
if 'audio_summary_path' not in st.session_state:
    st.session_state.audio_summary_path = None

# --- Sidebar ---
with st.sidebar:
    st.title("🔐 API Key & Settings")
    api_key_input = st.text_input("Enter Gemini API Key", type="password", key="api_key_input_widget")

    # Validate API Key Button
    if st.button("Validate API Key"):
        if api_key_input:
            try:
                genai.configure(api_key=api_key_input)
                # Test with a simple model listing or small generation
                st.session_state.gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                # Perform a minimal test to ensure the key/model works
                st.session_state.gemini_model.generate_content("test", request_options={'timeout': 10})
                st.session_state.api_key_validated = True
                st.success("API Key Validated Successfully!")
                st.rerun() # Rerun to update the main page state
            except Exception as e:
                st.session_state.api_key_validated = False
                st.session_state.gemini_model = None
                st.error(f"API Key Validation Failed: {e}")
        else:
            st.warning("Please enter an API Key.")

    if st.session_state.api_key_validated:
        st.success("Gemini API Key is Valid ✅")
    else:
        st.error("Gemini API Key Not Validated ❌")

    st.markdown("---")
    st.markdown("Built with [Streamlit](https://streamlit.io), [Gemini](https://ai.google.dev/), [Pytube](https://pytube.io), [Whisper](https://github.com/openai/whisper)")

# --- Caching Whisper Model ---
@st.cache_resource
def load_whisper_model():
    print("Loading Whisper model...") # Add print statement for debugging cache
    try:
        model = whisper.load_model("base")
        print("Whisper model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

whisper_model = load_whisper_model()

# --- Utility: Download & Transcribe Video ---
def download_and_transcribe(video_url):
    if not whisper_model:
        st.error("Whisper model not loaded. Cannot transcribe.")
        return None, None, "Whisper model failed to load."

    temp_audio_path = None
    try:
        st.info("Attempting to access YouTube video...")
        yt = YouTube(video_url)

        # Add basic checks
        # yt.check_availability() # This can sometimes throw errors too

        st.info(f"Accessing audio stream for '{yt.title}'...")
        # Filter for audio streams, prefer DASH audio if available (often higher quality)
        stream = yt.streams.filter(only_audio=True, adaptive=True, file_extension='mp4').order_by('abr').desc().first()
        if not stream: # Fallback to progressive audio streams if no DASH audio
             st.warning("DASH audio stream not found, falling back to progressive.")
             stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()

        if not stream:
             st.error("No suitable audio stream found for this video.")
             return None, yt.title, "No suitable audio stream found."


        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_audio_file:
            temp_audio_path = temp_audio_file.name

        st.info(f"Downloading audio to temporary file: {temp_audio_path}...")
        stream.download(filename=temp_audio_path)
        st.info("Download complete. Starting transcription...")

        # Transcribe using cached Whisper model
        result = whisper_model.transcribe(temp_audio_path, fp16=False) # fp16=False can improve stability on some systems
        st.info("Transcription complete.")
        return result["text"], yt.title, None # Return transcript, title, and no error

    except RegexMatchError:
         error_msg = "Pytube Regex Error: Could not parse video page. YouTube might have updated its structure. Try updating pytube (`pip install --upgrade pytube`)."
         st.error(error_msg)
         return None, "Error", error_msg
    except PytubeError as e:
         error_msg = f"Pytube Error: {e}. The video might be unavailable, age-restricted, or private."
         st.error(error_msg)
         return None, "Error", error_msg
    except Exception as e:
        # Catch other potential errors (network issues, file system errors, etc.)
        error_msg = f"An unexpected error occurred during download/transcription: {e}"
        st.error(error_msg)
        return None, "Error", error_msg
    finally:
        # Ensure temporary file is always cleaned up
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"Removed temp file: {temp_audio_path}")
            except OSError as e:
                st.warning(f"Could not remove temporary file {temp_audio_path}: {e}")


# --- Utility: Gemini Content Generator ---
def gemini_generate(text_input, prompt):
    if not st.session_state.api_key_validated or not st.session_state.gemini_model:
        return "Error: API Key not validated or Gemini model not initialized."

    full_prompt = f"{prompt}\n\nVideo Transcript:\n\"\"\"\n{text_input}\n\"\"\""
    try:
        # Add a timeout to prevent hanging indefinitely
        response = st.session_state.gemini_model.generate_content(full_prompt, request_options={'timeout': 180}) # 3 min timeout
        # Check for safety ratings if needed response.prompt_feedback
        if not response.parts:
             return "Error: Gemini returned an empty response. The content might have been blocked due to safety settings or the prompt was invalid."
        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        # Consider more specific error handling based on google.api_core.exceptions
        return f"Error generating content: {e}"

# --- Utility: TTS ---
def text_to_audio(text, filename=TTS_FILENAME):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        return filename, None # Return filename and no error
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None, f"Failed to generate audio: {e}"

# --- Main App Logic ---

st.subheader("Step 1: Provide Video URL")
url_input = st.text_input("Paste YouTube video URL:", value=st.session_state.video_url, key="url_input_widget")

# Update session state if URL changes
if url_input != st.session_state.video_url:
    st.session_state.video_url = url_input
    # Clear previous results when URL changes
    st.session_state.transcript = None
    st.session_state.video_title = None
    st.session_state.processing_error = None
    st.session_state.summary = None
    st.session_state.qa_answer = None
    st.session_state.flashcards = None
    st.session_state.quiz = None
    st.session_state.key_points = None
    st.session_state.audio_summary_path = None


# Process Button - Only proceed if URL is provided and API key is valid
if st.session_state.video_url and st.session_state.api_key_validated:
    if st.button("Process Video", key="process_button"):
        # Reset previous errors and results before processing
        st.session_state.processing_error = None
        st.session_state.transcript = None
        st.session_state.video_title = None

        with st.spinner("Downloading and transcribing video... This may take a few minutes."):
            transcript, title, error = download_and_transcribe(st.session_state.video_url)
            if error:
                st.session_state.processing_error = error
                st.session_state.transcript = None
                st.session_state.video_title = None
            else:
                st.session_state.transcript = transcript
                st.session_state.video_title = title
                st.session_state.processing_error = None # Clear error on success
                st.success(f"Video processed successfully: **{st.session_state.video_title}**")
                # Clear downstream results that depend on the transcript
                st.session_state.summary = None
                st.session_state.qa_answer = None
                st.session_state.flashcards = None
                st.session_state.quiz = None
                st.session_state.key_points = None
                st.session_state.audio_summary_path = None


elif not st.session_state.api_key_validated:
    st.warning("Please validate your Gemini API key in the sidebar.")
elif not st.session_state.video_url:
     st.info("Enter a YouTube video URL above and click 'Process Video'.")


# --- Display Results Section ---
# Only show results section if transcription was successful (no error and transcript exists)
if st.session_state.transcript and not st.session_state.processing_error:

    st.subheader("Step 2: AI-Powered Analysis")
    st.write(f"**Video Title:** {st.session_state.video_title}")

    tab_names = [
        "📝 Summary", "❓ Q&A", "💡 Flashcards", "🧐 Quiz", "🔑 Key Points",
        "📜 Transcript", "🔊 Audio Overview", "🙂 Sentiment", "📊 Topics", "💾 Export"
    ]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(tab_names)

    # Use the transcript stored in session state
    transcript_text = st.session_state.transcript

    with tab1: # Summary
        st.markdown("**Summary:**")
        if not st.session_state.summary: # Generate only if not already generated
             with st.spinner("Generating summary..."):
                 st.session_state.summary = gemini_generate(
                     transcript_text,
                     "Summarize this video's transcript concisely (2-4 paragraphs). Focus on the main topics, key findings, or conclusions. Aim for clarity and engagement."
                 )
        st.write(st.session_state.summary)

    with tab2: # Q&A
        st.markdown("**Ask a question about the video:**")
        question = st.text_input("Your question:", key="qa_question")
        if st.button("Get Answer", key="qa_button"):
            if question:
                with st.spinner("Thinking..."):
                    st.session_state.qa_answer = gemini_generate(
                        transcript_text,
                        f"Based *only* on the provided video transcript, answer the following question accurately and concisely:\n\nQuestion: {question}\n\nAnswer:"
                    )
            else:
                st.warning("Please enter a question.")
        # Display the answer if it exists in session state
        if st.session_state.qa_answer:
             st.write("**Answer:**", st.session_state.qa_answer)


    with tab3: # Flashcards
        st.markdown("**Flashcards (Question/Answer):**")
        if not st.session_state.flashcards: # Generate only once
             with st.spinner("Creating flashcards..."):
                 st.session_state.flashcards = gemini_generate(
                     transcript_text,
                     "Generate 5 flashcards based on the key information in this transcript. Format each as:\nQ: [Question related to a key point]\nA: [Concise answer based *only* on the transcript]\n\n---\n"
                 )
        st.code(st.session_state.flashcards, language="markdown")

    with tab4: # Quiz
        st.markdown("**Multiple Choice Quiz:**")
        if not st.session_state.quiz: # Generate only once
             with st.spinner("Building quiz..."):
                 st.session_state.quiz = gemini_generate(
                     transcript_text,
                     "Create a 5-question multiple-choice quiz based *only* on the provided transcript. Include 4 options (A, B, C, D) for each question and indicate the correct answer. Format clearly."
                 )
        st.code(st.session_state.quiz, language="markdown")

    with tab5: # Key Points
        st.markdown("**Key Points:**")
        if not st.session_state.key_points: # Generate only once
             with st.spinner("Extracting key points..."):
                  st.session_state.key_points = gemini_generate(
                      transcript_text,
                      "List the main key points or takeaways from this transcript as a bulleted list. Be concise and focus on the most important information."
                  )
        st.write(st.session_state.key_points) # Use st.write or st.markdown for bullet points

    with tab6: # Transcript
        st.markdown("**Full Transcript:**")
        # Consider adding timestamps if whisper provides them with higher accuracy models
        # For the 'base' model, timestamps can be less reliable.
        st.text_area("Transcript Text", transcript_text, height=300)


    with tab7: # Audio Overview
        st.markdown("**Voiceover Summary:**")
        # Generate audio only if summary exists and audio path isn't set
        if st.session_state.summary and not st.session_state.audio_summary_path:
            with st.spinner("Generating audio summary..."):
                 audio_path, audio_error = text_to_audio(st.session_state.summary, TTS_FILENAME)
                 if audio_error:
                     st.error(audio_error)
                     st.session_state.audio_summary_path = None
                 else:
                     st.session_state.audio_summary_path = audio_path

        # Display audio player if path is valid
        if st.session_state.audio_summary_path and os.path.exists(st.session_state.audio_summary_path):
             st.audio(st.session_state.audio_summary_path, format="audio/mp3")
        elif st.session_state.summary: # Show button if summary exists but audio failed/not generated
             st.warning("Could not generate or find audio file.")


    with tab8: # Sentiment (Placeholder)
        st.markdown("**Sentiment Analysis (coming soon...)**")
        st.info("This feature will analyze emotional tone throughout the video.")
        # Future implementation idea: Segment transcript, get sentiment per segment from Gemini

    with tab9: # Topics (Placeholder)
        st.markdown("**Topic Visualization (coming soon...)**")
        st.info("This will show topic shifts using visual graphs.")
        # Future implementation idea: Use Gemini to identify key topics, maybe visualize frequency or flow

    with tab10: # Export
        st.markdown("**Export Options:**")
        if st.session_state.summary:
            st.download_button(
                label="Download Summary (.txt)",
                data=st.session_state.summary,
                file_name=f"{st.session_state.video_title or 'video'}_summary.txt",
                mime="text/plain"
            )
        if st.session_state.transcript:
             st.download_button(
                 label="Download Transcript (.txt)",
                 data=st.session_state.transcript,
                 file_name=f"{st.session_state.video_title or 'video'}_transcript.txt",
                 mime="text/plain"
             )
        # Audio download needs careful handling in Streamlit
        if st.session_state.audio_summary_path and os.path.exists(st.session_state.audio_summary_path):
            try:
                with open(st.session_state.audio_summary_path, "rb") as fp:
                    st.download_button(
                        label="Download Audio Summary (.mp3)",
                        data=fp,
                        file_name=f"{st.session_state.video_title or 'video'}_summary.mp3",
                        mime="audio/mp3"
                    )
            except Exception as e:
                st.warning(f"Could not prepare audio for download: {e}")

# Display processing errors if they occurred
elif st.session_state.processing_error:
     st.error(f"Failed to process video: {st.session_state.processing_error}")
