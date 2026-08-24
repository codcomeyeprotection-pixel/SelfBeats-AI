import os
import streamlit as st
from gradio_client import Client

st.set_page_config(page_title="SelfBeats AI", page_icon="🎵", layout="centered")

st.title("🎵 SelfBeats AI")
st.subheader("Create Your Own Copyright-Free Music for Reels & Shorts")

platform = st.radio("Select Platform", ["Instagram Reels", "YouTube Shorts"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    genre = st.selectbox("Music Style", ["Phonk Drop", "Lo-Fi Chill Beat", "Cinematic Hype", "Upbeat Vlog", "Dark Ambient"])
with col2:
    mood = st.selectbox("Mood", ["Energetic", "Relaxing", "Motivational", "Aggressive", "Mysterious"])

if st.button("🚀 Generate My Own Music", use_container_width=True):
    with st.spinner("⚡ Connecting to AI Engine... Generating track (takes 25-35 seconds)..."):
        prompt_text = f"A high quality {mood.lower()} {genre.lower()} background music track for {platform}. Catchy rhythm, heavy phonk beats."
        
        try:
            # Auto-routing through Gradio Space
            client = Client("facebook/MusicGen")
            result = client.predict(
                "facebook/musicgen-small",  # Model name parameter
                prompt_text,                 # Text prompt
                None,                        # Audio input (empty)
                10                           # Duration in seconds
            )
            
            # Read generated audio file path
            audio_path = result if isinstance(result, str) else result[0]
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
                
            st.success("🎉 Music Generated Successfully!")
            st.audio(audio_bytes, format="audio/wav")
            st.download_button(
                label="⬇️ Download Track (.wav)",
                data=audio_bytes,
                file_name=f"SelfBeats_{genre.replace(' ', '_')}.wav",
                mime="audio/wav",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"⚠️ Generation error: {e}. Please tap 'Generate' again in 5 seconds.")