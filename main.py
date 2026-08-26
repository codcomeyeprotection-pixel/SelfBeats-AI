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

def generate_audio_with_fallback(prompt, token):
    # Method 1: Hugging Face Inference API via requests (with wait_for_model)
    headers = {"Authorization": f"Bearer {token}"}
    urls = [
        "https://router.huggingface.co/hf-inference/models/facebook/musicgen-small",
        "https://api-inference.huggingface.co/models/facebook/musicgen-small"
    ]
    
    for url in urls:
        for attempt in range(3):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json={"inputs": prompt, "options": {"wait_for_model": True}}, 
                    timeout=90
                )
                if response.status_code == 200 and len(response.content) > 1000:
                    return response.content, None
                elif response.status_code == 503:
                    time.sleep(10) # Model loading GPU
            except Exception:
                pass
    return None, "Server busy or endpoint unavailable. Please try again in a few seconds."

if st.button("🚀 Generate My Own Music", use_container_width=True):
    if not HF_TOKEN:
        st.error("⚠️ Secrets mein 'HF_TOKEN' miss ho gaya hai! Secrets check karein.")
    else:
        with st.spinner("⚡ Connecting to AI Engine... Generating track (20-30 seconds)..."):
            prompt_text = f"A high quality {mood.lower()} {genre.lower()} background music track for {platform}. Catchy rhythm, heavy phonk beats."
            
            audio_bytes, error_msg = generate_audio_with_fallback(prompt_text, HF_TOKEN)
            
            if audio_bytes:
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
                st.error(f"⚠️ {error_msg}")