import streamlit as st
import tempfile
import os
import base64
import google.generativeai as genai
from gtts import gTTS
from pytube import YouTube
from pytube.exceptions import PytubeError, RegexMatchError
import re # Import regular expressions for cleaning SRT
import time

# --- Constants ---
TTS_FILENAME = "summary_audio.mp3"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

# --- Page Config ---
st.set_page_config(page_title="Video Summarizer Pro", layout="wide")
st.title("🎬 Video Summarizer Pro (Caption-Based)")
st.markdown("Summarize YouTube videos using their captions, ask questions, generate quizzes, and more!")
st.info("Ensure you have the latest `pytube` installed: `pip install --upgrade pytube`")

# --- Session State Initialization ---
# (Keep session state initialization as before, maybe rename transcript keys)
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'video_url' not in st.session_state:
    st.session_state.video_url = ""
# Rename transcript related state
if 'caption_text' not in st.session_state:
    st.session_state.caption_text = None
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

    if st.button("Validate API Key"):
        if api_key_input:
            try:
                genai.configure(api_key=api_key_input)
                st.session_state.gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                # Perform a minimal test
                st.session_state.gemini_model.generate_content("test", request_options={'timeout': 10})
                st.session_state.api_key_validated = True
                st.success("API Key Validated Successfully!")
                # No rerun needed here, state is set
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
    st.markdown("Built with [Streamlit](https://streamlit.io), [Gemini](https://ai.google.dev/), [Pytube](https://pytube.io), [gTTS](https://github.com/pndurette/gTTS)")

# --- REMOVED Whisper Model Loading ---

# --- Utility: Get Captions & Title ---
def clean_srt(srt_text):
    """Removes timestamps and formatting from SRT captions."""
    # Remove index numbers
    text = re.sub(r'^\d+\s*$', '', srt_text, flags=re.MULTILINE)
    # Remove timestamps
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\s*$', '', text, flags=re.MULTILINE)
    # Remove potential HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove extra blank lines
    text = re.sub(r'\n\s*\n', '\n', text, flags=re.MULTILINE).strip()
    return text

def get_captions_and_title(video_url, lang_code='en'):
    """Fetches video title and captions directly from YouTube."""
    try:
        st.info("Attempting to access YouTube video...")
        yt = YouTube(video_url)
        title = yt.title
        st.info(f"Video Title: '{title}'")
        st.info(f"Looking for captions in language: '{lang_code}'...")

        # List available captions
        available_captions = yt.captions
        if not available_captions:
             st.warning("No caption tracks found for this video.")
             return None, title, f"No caption tracks found for '{title}'."

        # Try to get the specified language caption
        caption = available_captions.get(lang_code)

        # Fallback: Try auto-generated captions if specific language not found
        if not caption and f'a.{lang_code}' in available_captions:
            st.info(f"'{lang_code}' captions not found, trying auto-generated ('a.{lang_code}')...")
            caption = available_captions.get(f'a.{lang_code}')

        # Fallback: Get the first available caption if still none found
        if not caption:
             first_caption_code = list(available_captions.keys())[0]
             st.warning(f"Neither '{lang_code}' nor 'a.{lang_code}' captions found. Using the first available: '{first_caption_code}'")
             caption = available_captions.get(first_caption_code)

        if not caption:
            # This should ideally not happen if available_captions was not empty, but as a safeguard
             st.error("Could not retrieve any caption track.")
             return None, title, f"Failed to retrieve any captions for '{title}'."

        st.info(f"Fetching '{caption.code}' captions...")
        # Generate SRT (text format with timestamps)
        srt_captions = caption.generate_srt_captions()

        if not srt_captions:
             st.error("Caption track found, but failed to generate SRT content.")
             return None, title, f"Failed to generate SRT content for '{caption.code}' captions."

        st.info("Cleaning captions...")
        plain_text_captions = clean_srt(srt_captions)

        st.info("Caption processing complete.")
        return plain_text_captions, title, None # Return captions, title, and no error

    except RegexMatchError:
         error_msg = "Pytube Regex Error: Could not parse video page. YouTube might have updated its structure. Try updating pytube (`pip install --upgrade pytube`)."
         st.error(error_msg)
         return None, "Error", error_msg
    except PytubeError as e:
         error_msg = f"Pytube Error: {e}. The video might be unavailable, age-restricted, or private."
         st.error(error_msg)
         return None, "Error", error_msg
    except Exception as e:
        # Catch other potential errors (network issues, etc.)
        error_msg = f"An unexpected error occurred: {e}"
        st.error(error_msg)
        return None, "Error", error_msg
    # No temporary file cleanup needed anymore

# --- Utility: Gemini Content Generator ---
# (Keep gemini_generate as before)
def gemini_generate(text_input, prompt):
    if not st.session_state.api_key_validated or not st.session_state.gemini_model:
        return "Error: API Key not validated or Gemini model not initialized."

    # Updated prompt slightly for clarity
    full_prompt = f"{prompt}\n\nVideo Captions:\n\"\"\"\n{text_input}\n\"\"\""
    try:
        response = st.session_state.gemini_model.generate_content(full_prompt, request_options={'timeout': 180})
        if not response.parts:
             # Check for blocked content due to safety
             if response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 return f"Error: Gemini request blocked due to safety settings ({block_reason})."
             else:
                 return "Error: Gemini returned an empty response. The prompt might be invalid or the model could not generate content."
        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return f"Error generating content: {e}"


# --- Utility: TTS ---
# (Keep text_to_audio as before)
def text_to_audio(text, filename=TTS_FILENAME):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        # Clear previous audio path before generating new one
        if 'audio_summary_path' in st.session_state:
             if st.session_state.audio_summary_path and os.path.exists(st.session_state.audio_summary_path):
                 try:
                      os.remove(st.session_state.audio_summary_path)
                 except OSError:
                      pass # Ignore if removal fails
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
    st.session_state.caption_text = None # Clear caption text
    st.session_state.video_title = None
    st.session_state.processing_error = None
    st.session_state.summary = None
    st.session_state.qa_answer = None
    st.session_state.flashcards = None
    st.session_state.quiz = None
    st.session_state.key_points = None
    st.session_state.audio_summary_path = None


# Process Button
if st.session_state.video_url and st.session_state.api_key_validated:
    if st.button("Process Video Captions", key="process_button"):
        # Reset state before processing
        st.session_state.processing_error = None
        st.session_state.caption_text = None # Reset caption text
        st.session_state.video_title = None
        # Clear downstream results
        st.session_state.summary = None
        st.session_state.qa_answer = None
        st.session_state.flashcards = None
        st.session_state.quiz = None
        st.session_state.key_points = None
        st.session_state.audio_summary_path = None


        with st.spinner("Fetching and processing video captions..."):
            # Call the updated function
            captions, title, error = get_captions_and_title(st.session_state.video_url)
            if error:
                st.session_state.processing_error = error
                st.session_state.caption_text = None # Ensure caption is None on error
                st.session_state.video_title = title # Keep title if available even on error
            elif not captions: # Handle case where function returns None for captions without error msg
                 st.session_state.processing_error = "No caption text could be retrieved, although no specific error was raised."
                 st.session_state.caption_text = None
                 st.session_state.video_title = title
            else:
                st.session_state.caption_text = captions # Store caption text
                st.session_state.video_title = title
                st.session_state.processing_error = None # Clear error on success
                st.success(f"Video captions processed successfully for: **{st.session_state.video_title}**")

elif not st.session_state.api_key_validated:
    st.warning("Please validate your Gemini API key in the sidebar.")
elif not st.session_state.video_url:
     st.info("Enter a YouTube video URL above and click 'Process Video Captions'.")


# --- Display Results Section ---
# Only show results if captions were successfully fetched
if st.session_state.caption_text and not st.session_state.processing_error:

    st.subheader("Step 2: AI-Powered Analysis from Captions")
    st.write(f"**Video Title:** {st.session_state.video_title}")

    tab_names = [
        "📝 Summary", "❓ Q&A", "💡 Flashcards", "🧐 Quiz", "🔑 Key Points",
        "📜 Captions", "🔊 Audio Overview", "🙂 Sentiment", "📊 Topics", "💾 Export" # Changed "Transcript" to "Captions"
    ]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(tab_names)

    # Use the caption text stored in session state
    caption_content = st.session_state.caption_text

    with tab1: # Summary
        st.markdown("**Summary:**")
        if not st.session_state.summary:
             with st.spinner("Generating summary from captions..."):
                 st.session_state.summary = gemini_generate(
                     caption_content,
                     "Summarize the key information found in these video captions concisely (2-4 paragraphs). Focus on the main topics, key findings, or conclusions. Assume the captions represent the video's content."
                 )
        st.write(st.session_state.summary)

    with tab2: # Q&A
        st.markdown("**Ask a question based on the captions:**")
        question = st.text_input("Your question:", key="qa_question")
        if st.button("Get Answer", key="qa_button"):
            if question:
                with st.spinner("Searching captions for answer..."):
                    st.session_state.qa_answer = gemini_generate(
                        caption_content,
                        f"Based *only* on the provided video captions, answer the following question accurately and concisely:\n\nQuestion: {question}\n\nAnswer:"
                    )
            else:
                st.warning("Please enter a question.")
        if st.session_state.qa_answer:
             st.write("**Answer:**", st.session_state.qa_answer)


    with tab3: # Flashcards
        st.markdown("**Flashcards (Based on Captions):**")
        if not st.session_state.flashcards:
             with st.spinner("Creating flashcards from captions..."):
                 st.session_state.flashcards = gemini_generate(
                     caption_content,
                     "Generate 5 flashcards based on the key information in these video captions. Format each as:\nQ: [Question related to a key point]\nA: [Concise answer based *only* on the captions]\n\n---\n"
                 )
        st.code(st.session_state.flashcards, language="markdown")

    with tab4: # Quiz
        st.markdown("**Multiple Choice Quiz (Based on Captions):**")
        if not st.session_state.quiz:
             with st.spinner("Building quiz from captions..."):
                 st.session_state.quiz = gemini_generate(
                     caption_content,
                     "Create a 5-question multiple-choice quiz based *only* on the provided video captions. Include 4 options (A, B, C, D) for each question and indicate the correct answer. Format clearly."
                 )
        st.code(st.session_state.quiz, language="markdown")

    with tab5: # Key Points
        st.markdown("**Key Points (from Captions):**")
        if not st.session_state.key_points:
             with st.spinner("Extracting key points from captions..."):
                  st.session_state.key_points = gemini_generate(
                      caption_content,
                      "List the main key points or takeaways from these video captions as a bulleted list. Be concise and focus on the most important information presented in the text."
                  )
        st.write(st.session_state.key_points)

    with tab6: # Captions Display
        st.markdown("**Video Captions Text:**")
        st.text_area("Caption Content", caption_content, height=300)


    with tab7: # Audio Overview
        st.markdown("**Voiceover Summary:**")
        if st.session_state.summary and not st.session_state.audio_summary_path:
            # Check if the audio file still exists from a previous run in this session
            # Re-generate only if needed
            if not os.path.exists(TTS_FILENAME):
                 with st.spinner("Generating audio summary..."):
                     audio_path, audio_error = text_to_audio(st.session_state.summary, TTS_FILENAME)
                     if audio_error:
                         st.error(audio_error)
                         st.session_state.audio_summary_path = None
                     else:
                         st.session_state.audio_summary_path = audio_path
            else:
                 st.session_state.audio_summary_path = TTS_FILENAME # Use existing file

        if st.session_state.audio_summary_path and os.path.exists(st.session_state.audio_summary_path):
             st.audio(st.session_state.audio_summary_path, format="audio/mp3")
        elif st.session_state.summary:
             st.warning("Could not generate or find audio file.")


    with tab8: # Sentiment (Placeholder)
        st.markdown("**Sentiment Analysis (coming soon...)**")
        st.info("This feature will analyze emotional tone based on the captions.")
        # Future: Use Gemini to analyze sentiment of caption_content

    with tab9: # Topics (Placeholder)
        st.markdown("**Topic Visualization (coming soon...)**")
        st.info("This will show topic shifts based on caption analysis.")
        # Future: Use Gemini to identify topics in caption_content

    with tab10: # Export
        st.markdown("**Export Options:**")
        # Use caption_content for download data
        if st.session_state.summary:
            st.download_button(
                label="Download Summary (.txt)",
                data=st.session_state.summary,
                file_name=f"{st.session_state.video_title or 'video'}_summary.txt",
                mime="text/plain"
            )
        if st.session_state.caption_text: # Check caption_text for download
             st.download_button(
                 label="Download Captions (.txt)", # Changed label
                 data=st.session_state.caption_text, # Use caption_text
                 file_name=f"{st.session_state.video_title or 'video'}_captions.txt", # Changed filename
                 mime="text/plain"
             )
        if st.session_state.audio_summary_path and os.path.exists(st.session_state.audio_summary_path):
            try:
                with open(st.session_state.audio_summary_path, "rb") as fp:
                    st.download_button(
                        label="Download Audio Summary (.mp3)",
                        data=fp, # Read bytes directly
                        file_name=f"{st.session_state.video_title or 'video'}_summary.mp3",
                        mime="audio/mp3"
                    )
            except Exception as e:
                st.warning(f"Could not prepare audio for download: {e}")

# Display processing errors if they occurred
elif st.session_state.processing_error:
     st.error(f"Failed to process video: {st.session_state.processing_error}")

# Add cleanup for audio file on exit? Maybe not necessary for temp streamlit runs.
