import os
import requests
import streamlit as st

st.set_page_config(page_title="SelfBeats AI", page_icon="🎵", layout="centered")

st.title("🎵 SelfBeats AI")
st.subheader("Create Your Own Copyright-Free Music for Reels & Shorts")

HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/facebook/musicgen-small"

platform = st.radio("Select Platform", ["Instagram Reels", "YouTube Shorts"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    genre = st.selectbox("Music Style", ["Phonk Drop", "Lo-Fi Chill Beat", "Cinematic Hype", "Upbeat Vlog", "Dark Ambient"])
with col2:
    mood = st.selectbox("Mood", ["Energetic", "Relaxing", "Motivational", "Aggressive", "Mysterious"])

if st.button("🚀 Generate My Own Music", use_container_width=True):
    if not HF_TOKEN:
        st.error("⚠️ Secrets mein 'HF_TOKEN' miss ho gaya hai!")
    else:
        st.info("⚡ Generating Music... (Takes 15-30 seconds)")
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        prompt_text = f"A high quality {mood.lower()} {genre.lower()} background music track designed for {platform}. Catchy rhythm, loopable."
        
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt_text}, timeout=120)
            if response.status_code == 200:
                audio_bytes = response.content
                st.success("🎉 Music Generated Successfully!")
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    label="⬇️ Download Track (.wav)",
                    data=audio_bytes,
                    file_name=f"SelfBeats_{genre.replace(' ', '_')}.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
            else:
                st.warning("⚠️ Open-Source Model Warm-Up me hai. 10 sec baad dobara try karein.")
        except Exception as e:
            st.error(f"Error: {e}")