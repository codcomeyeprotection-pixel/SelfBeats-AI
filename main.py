import os
import time
import requests
import streamlit as st

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
    if not HF_TOKEN:
        st.error("⚠️ Secrets mein 'HF_TOKEN' missing hai!")
    else:
        with st.spinner("⚡ Synthesizing Beat & Rhythm... Please wait 15-25 seconds..."):
            prompt_text = f"Phonk bass drop beat, high energy background music for {platform}, {genre}, {mood} style, instrumental 128bpm loop"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            
            audio_data = None
            for model_url in MODELS:
                try:
                    res = requests.post(
                        model_url,
                        headers=headers,
                        json={"inputs": prompt_text, "options": {"wait_for_model": True}},
                        timeout=60
                    )
                    if res.status_code == 200 and len(res.content) > 5000:
                        audio_data = res.content
                        break
                except Exception:
                    continue

            if audio_data:
                st.success("🎉 Music Generated Successfully!")
                st.audio(audio_data, format="audio/wav")
                st.download_button(
                    label="⬇️ Download Track (.wav)",
                    data=audio_data,
                    file_name=f"SelfBeats_{genre.replace(' ', '_')}.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
            else:
                st.error("⚠️ HuggingFace Free Tier is temporarily overloaded. Please tap 'Generate' once again in 10 seconds.")