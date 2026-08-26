import io
import numpy as np
import scipy.io.wavfile as wav
import streamlit as st

st.set_page_config(page_title="SelfBeats AI", page_icon="🎵", layout="centered")

st.title("🎵 SelfBeats AI")
st.subheader("Create Your Own Copyright-Free Music for Reels & Shorts")

platform = st.radio("Select Platform", ["Instagram Reels", "YouTube Shorts"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    genre = st.selectbox("Music Style", ["Phonk Drop", "Lo-Fi Chill Beat", "Cinematic Hype", "Upbeat Vlog", "Dark Ambient"])
with col2:
    mood = st.selectbox("Mood", ["Energetic", "Relaxing", "Motivational", "Aggressive", "Mysterious"])

def generate_pro_beat(genre, duration=10, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Base kick drum rhythm
    bpm = 130 if "Phonk" in genre or "Hype" in genre else 85
    beat_freq = bpm / 60.0
    kick = np.sin(2 * np.pi * 50 * t) * np.exp(-5 * (t % (1 / beat_freq)))
    
    # Synth melody layer
    if "Phonk" in genre:
        freq = 150 + 50 * np.sin(2 * np.pi * 2 * t)
        synth = 0.6 * np.sin(2 * np.pi * freq * t) + 0.4 * (np.random.rand(len(t)) - 0.5)
    elif "Lo-Fi" in genre:
        synth = 0.5 * np.sin(2 * np.pi * 220 * t) * np.exp(-2 * (t % 1))
    else:
        synth = 0.5 * np.sin(2 * np.pi * 330 * t)
        
    # Mix audio layers
    audio_signal = kick * 0.6 + synth * 0.4
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    
    # Convert to 16-bit PCM WAV
    audio_int16 = (audio_signal * 32767).astype(np.int16)
    
    byte_io = io.BytesIO()
    wav.write(byte_io, sample_rate, audio_int16)
    return byte_io.getvalue()

if st.button("🚀 Generate My Own Music", use_container_width=True):
    with st.spinner("⚡ Synthesizing Track... Generating 100% Copyright-Free Beats..."):
        audio_bytes = generate_pro_beat(genre)
        
        st.success("🎉 Music Track Generated Instantly!")
        st.audio(audio_bytes, format="audio/wav")
        st.download_button(
            label="⬇️ Download Track (.wav)",
            data=audio_bytes,
            file_name=f"SelfBeats_{genre.replace(' ', '_')}.wav",
            mime="audio/wav",
            use_container_width=True
        )