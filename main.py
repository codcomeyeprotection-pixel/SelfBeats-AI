import os
import time
import requests
import streamlit as st

st.set_page_config(page_title="SelfBeats AI", page_icon="🎵", layout="centered")

st.title("🎵 SelfBeats AI")
st.subheader("Create Your Own Copyright-Free Music for Reels & Shorts")

HF_TOKEN = os.environ.get("HF_TOKEN")
# Updated working endpoints
PRIMARY_URL = "https://router.huggingface.co/hf-inference/models/facebook/musicgen-small"
FALLBACK_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"

platform = st.radio("Select Platform", ["Instagram Reels", "YouTube Shorts"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    genre = st.selectbox("Music Style", ["Phonk Drop", "Lo-Fi Chill Beat", "Cinematic Hype", "Upbeat Vlog", "Dark Ambient"])
with col2:
    mood = st.selectbox("Mood", ["Energetic", "Relaxing", "Motivational", "Aggressive", "Mysterious"])

if st.button("🚀 Generate My Own Music", use_container_width=True):
    if not HF_TOKEN:
        st.error("⚠️ Secrets mein 'HF_TOKEN' missing hai. Please update HF_TOKEN in Secrets.")
    else:
        st.info("⚡ Generating Track... Connecting to AI Model (takes 20-40 seconds)...")
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        prompt_text = f"A high quality {mood.lower()} {genre.lower()} background music track for {platform}. Catchy rhythm, loopable, heavy phonk beats."
        
        success = False
        endpoints = [PRIMARY_URL, FALLBACK_URL]
        
        for url in endpoints:
            if success:
                break
            for attempt in range(3):
                try:
                    response = requests.post(
                        url, 
                        headers=headers, 
                        json={"inputs": prompt_text, "options": {"wait_for_model": True}}, 
                        timeout=90
                    )
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
                        st.warning(f"⏳ GPU Warm-up in progress... Attempt {attempt+1}/3... Waiting 10s.")
                        time.sleep(10)
                    else:
                        break
                except Exception as e:
                    time.sleep(3)
                    continue

        if not success:
            st.error("❌ Connection Timeout. Server temporarily busy, please tap 'Generate' again in 10 seconds.")