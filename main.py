import os
import streamlit as st
from gradio_client import Client

st.set_page_config(page_title="SelfBeats AI", page_icon="🎵", layout="centered")

st.title("🎵 SelfBeats AI")
st.subheader("Create Your Own Copyright-Free Music for Reels & Shorts")

HF_TOKEN = os.environ.get("HF_TOKEN")

platform = st.radio("Select Platform", ["Instagram Reels", "YouTube Shorts"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    genre = st.selectbox("Music Style", ["Phonk Drop", "Lo-Fi Chill Beat", "Cinematic Hype", "Upbeat Vlog", "Dark Ambient"])
with col2:
    mood = st.selectbox("Mood", ["Energetic", "Relaxing", "Motivational", "Aggressive", "Mysterious"])

if st.button("🚀 Generate My Own Music", use_container_width=True):
    with st.spinner("⚡ Connecting to AI Queue... Track generating in 20-30 seconds..."):
        prompt_text = f"A high quality {mood.lower()} {genre.lower()} background music track for {platform}. Catchy rhythm, loopable, heavy phonk beats."
        
        try:
            # Connects directly to HuggingFace MusicGen Space Queue
            client = Client("facebook/MusicGen")
            result = client.predict(
                model="facebook/musicgen-small",
                text_prompt=prompt_text,
                duration=10,
                api_name="/predict"
            )
            
            with open(result, "rb") as f:
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
            st.error(f"Server busy, retrying queue. Please tap 'Generate' again: {e}")