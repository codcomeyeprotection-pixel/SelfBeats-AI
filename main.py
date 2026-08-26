import io
import time
import random
import numpy as np
import scipy.io.wavfile as wav
import streamlit as st

st.set_page_config(page_title="SelfBeats AI", page_icon="🎵", layout="centered")

st.title("🎵 SelfBeats AI")
st.subheader("Create Your Own Copyright-Free Music for Reels & Shorts")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    platform = st.selectbox("Platform", ["Instagram Reels", "YouTube Shorts"])
with col2:
    genre = st.selectbox("Music Style", [
        "Phonk Drop", "Lo-Fi Chill Beat", "Cinematic Hype", 
        "Cyberpunk Drift", "Trap Bass", "Ambient Drone", "Infinite Random Beat"
    ])
with col3:
    mood = st.selectbox("Mood", [
        "Energetic", "Dark / Aggressive", "Relaxing", 
        "Motivational", "Mysterious", "Ethereal"
    ])

def get_musical_frequencies(scale="minor"):
    # Frequencies for A Minor / Pentatonic Scale for natural human feel
    scales = {
        "minor": [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00],
        "phonk": [110.00, 123.47, 130.81, 146.83, 164.81, 174.61, 196.00, 220.00],
        "lofi": [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
    }
    return scales.get(scale, scales["minor"])

def generate_human_type_beat(genre_type, mood_type, duration=10, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    total_samples = len(t)
    audio = np.zeros(total_samples)
    
    # Random seed based on time for infinite dynamic variations
    seed = int(time.time() * 1000) % 100000
    random.seed(seed)
    np.random.seed(seed)
    
    # Determine BPM & Scale
    bpm = random.randint(130, 160) if "Phonk" in genre_type or "Trap" in genre_type else random.randint(75, 95)
    beat_sec = 60.0 / bpm
    notes = get_musical_frequencies("phonk" if "Phonk" in genre_type else "lofi" if "Lo-Fi" in genre_type else "minor")
    
    # Layer 1: Dynamic Kick / 808 Bass
    kick_interval = int(beat_sec * sample_rate)
    for i in range(0, total_samples, kick_interval):
        sub_len = min(int(0.2 * sample_rate), total_samples - i)
        sub_t = np.linspace(0, 0.2, sub_len, False)
        # Pitch drop kick drum
        freq_env = 120 * np.exp(-30 * sub_t) + 40
        kick_wave = np.sin(2 * np.pi * freq_env * sub_t) * np.exp(-10 * sub_t)
        audio[i:i+sub_len] += kick_wave * 0.8
        
    # Layer 2: Human-style Melody & Chords
    note_step = int((beat_sec / 2) * sample_rate) # 8th notes
    for i in range(0, total_samples, note_step):
        if random.random() > 0.2: # Natural human pause pattern
            note_freq = random.choice(notes)
            note_len = min(int((beat_sec) * sample_rate), total_samples - i)
            note_t = np.linspace(0, beat_sec, note_len, False)
            
            # Harmonic richness (Overtones)
            melody = 0.5 * np.sin(2 * np.pi * note_freq * note_t)
            melody += 0.25 * np.sin(2 * np.pi * (note_freq * 2) * note_t)
            melody *= np.exp(-3 * note_t) # Envelope decay
            
            audio[i:i+note_len] += melody * 0.4
            
    # Layer 3: Hi-Hats / Shaker
    hat_step = int((beat_sec / 4) * sample_rate)
    for i in range(0, total_samples, hat_step):
        hat_len = min(int(0.05 * sample_rate), total_samples - i)
        noise = (np.random.rand(hat_len) - 0.5) * np.exp(-50 * np.linspace(0, 0.05, hat_len, False))
        audio[i:i+hat_len] += noise * 0.15

    # Normalize Audio
    audio = audio / (np.max(np.abs(audio)) + 1e-5)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    byte_io = io.BytesIO()
    wav.write(byte_io, sample_rate, audio_int16)
    return byte_io.getvalue()

if st.button("🚀 Generate My Own Music", use_container_width=True):
    with st.spinner("⚡ Composing Dynamic Musical Beat... Please wait 3 seconds..."):
        audio_bytes = generate_human_type_beat(genre, mood)
        
        st.success(f"🎉 New Unique Beat Generated! ({genre} - {mood})")
        st.audio(audio_bytes, format="audio/wav")
        st.download_button(
            label="⬇️ Download Track (.wav)",
            data=audio_bytes,
            file_name=f"SelfBeats_{genre.replace(' ', '_')}_{platform.replace(' ', '')}.wav",
            mime="audio/wav",
            use_container_width=True
        )