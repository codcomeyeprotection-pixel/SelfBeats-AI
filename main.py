import os
import time
import requests
import streamlit as st

st.set_page_config(page_title="SelfBeats AI", page_icon="🎵", layout="centered")

st.title("🎵 SelfBeats AI")
st.subheader("Create Your Own Copyright-Free Music for Reels & Shorts")

HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"

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
        st.info("⚡ Generating Track... Initializing GPU...")
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        prompt_text = f"A high quality {mood.lower()} {genre.lower()} background music track for {platform}. Catchy rhythm, loopable."
        payload = {"inputs": prompt_text}
        
        success = False
        for attempt in range(5):
            try:
                response = requests.post("https://api-inference.huggingface.co/models/facebook/musicgen-small", headers=headers, json=payload, timeout=120)
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
                    success = True
                    break
                elif response.status_code in [503, 500, 429]:
                    st.warning(f"⏳ Model GPU par load ho raha hai... Retrying ({attempt+1}/5)... Wait karein.")
                    time.sleep(12)
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    break
            except Exception as e:
                st.error(f"Network Error: {e}")
                break