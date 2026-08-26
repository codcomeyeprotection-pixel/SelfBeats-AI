import io
import time
import random
import numpy as np
import scipy.io.wavfile as wav
import streamlit as st

st.set_page_config(page_title="SelfBeats AI", page_icon="🎵", layout="centered")

st.title("🎵 SelfBeats AI")
st.subheader("Create Your Own Copyright-Free Music for Reels & Shorts")

col1, col2 = st.columns(2)
with col1:
    platform = st.selectbox("Select Platform", ["Instagram Reels", "YouTube Shorts"])
with col2:
    genre = st.selectbox("Music Style", [
        "Phonk Drop", "Lo-Fi Chill Beat", "Cinematic Hype", 
        "Cyberpunk Drift", "Trap Bass", "Ambient Drone", 
        "Synthwave", "Infinite Random Beat"
    ])

mood = st.selectbox("Mood", [
    "Energetic", "Dark / Aggressive", "Relaxing", 
    "Motivational", "Mysterious", "Ethereal", "Sad / Melancholic"
])

def get_scale_notes(genre, mood):
    # Dynamic scale mapping for realistic human music feel
    if "Phonk" in genre or "Dark" in mood:
        base_freq = random.choice([110.0, 123.47, 130.81]) # Low Phonk root (A2, B2, C3)
        intervals = [0, 3, 5, 6, 7, 10, 12] # Minor Pentatonic / Blues scale
    elif "Lo-Fi" in genre or "Relaxing" in mood or "Sad" in mood:
        base_freq = random.choice([220.0, 261.63, 293.66]) # A3, C4, D4
        intervals = [0, 2, 4, 7, 9, 11, 12] # Major 7th / Jazz scale
    elif "Cinematic" in genre or "Motivational" in mood:
        base_freq = random.choice([146.83, 164.81, 196.00]) # D3, E3, G3
        intervals = [0, 2, 3, 5, 7, 8, 10, 12] # Aeolian / Epic scale
    else:
        base_freq = random.choice([174.61, 196.00, 220.00])
        intervals = [0, 2, 4, 5, 7, 9, 11, 12] # Diatonic scale

    # Convert semitones to frequencies
    return [base_freq * (2 ** (i / 12.0)) for i in intervals]

def generate_infinite_track(genre, mood, duration=12, sample_rate=44100):
    # Unique seed per click and per user environment
    user_unique_seed = int(time.time() * 1000) ^ random.randint(1000, 999999)
    random.seed(user_unique_seed)
    np.random.seed(user_unique_seed % (2**32 - 1))
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    total_samples = len(t)
    audio = np.zeros(total_samples)
    
    # 1. Dynamic BPM Selection based on Genre
    if "Phonk" in genre or "Trap" in genre:
        bpm = random.randint(135, 165)
    elif "Lo-Fi" in genre or "Relaxing" in mood:
        bpm = random.randint(70, 90)
    elif "Cinematic" in genre:
        bpm = random.randint(95, 120)
    else:
        bpm = random.randint(110, 140)
        
    beat_sec = 60.0 / bpm
    scale_notes = get_scale_notes(genre, mood)
    
    # 2. Kick / 808 Bass Line Generator
    kick_step = int(beat_sec * sample_rate)
    kick_pattern = random.choice([[1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 1, 1]])
    
    p_idx = 0
    for i in range(0, total_samples, kick_step):
        if kick_pattern[p_idx % len(kick_pattern)] == 1:
            k_len = min(int(0.25 * sample_rate), total_samples - i)
            k_t = np.linspace(0, 0.25, k_len, False)
            # Pitch-drop 808 sub kick
            freq_env = (140 if "Phonk" in genre else 100) * np.exp(-35 * k_t) + 35
            kick_wave = np.sin(2 * np.pi * freq_env * k_t) * np.exp(-8 * k_t)
            audio[i:i+k_len] += kick_wave * 0.85
        p_idx += 1

    # 3. Snare / Clap Layer
    snare_offset = int(beat_sec * sample_rate)
    for i in range(snare_offset, total_samples, snare_offset * 2):
        s_len = min(int(0.15 * sample_rate), total_samples - i)
        s_t = np.linspace(0, 0.15, s_len, False)
        snare_noise = (np.random.rand(s_len) - 0.5) * np.exp(-25 * s_t)
        snare_tone = np.sin(2 * np.pi * 180 * s_t) * np.exp(-30 * s_t)
        audio[i:i+s_len] += (snare_noise * 0.4 + snare_tone * 0.3)

    # 4. Humanized Melody & Chords (Micro-timing jitter & Soft Envelopes)
    note_step = int((beat_sec / 2) * sample_rate)
    for i in range(0, total_samples, note_step):
        if random.random() > 0.25:
            # Human Micro-timing offset
            human_jitter = random.randint(-100, 100)
            start_pos = max(0, min(total_samples - 1, i + human_jitter))
            
            note_freq = random.choice(scale_notes)
            m_len = min(int((beat_sec * random.choice([0.5, 1.0, 1.5])) * sample_rate), total_samples - start_pos)
            m_t = np.linspace(0, m_len / sample_rate, m_len, False)
            
            # Rich Saw/Sine Harmonics for Lead Melody
            lead = 0.4 * np.sin(2 * np.pi * note_freq * m_t)
            lead += 0.2 * np.sin(2 * np.pi * (note_freq * 2) * m_t) # Harmonic octave
            lead *= np.exp(-4 * m_t) # Natural decay envelope
            
            audio[start_pos:start_pos+m_len] += lead * 0.45

    # 5. Hi-Hat Rhythms (16th notes)
    hat_step = int((beat_sec / 4) * sample_rate)
    for i in range(0, total_samples, hat_step):
        if random.random() > 0.15:
            h_len = min(int(0.04 * sample_rate), total_samples - i)
            h_t = np.linspace(0, 0.04, h_len, False)
            hat_noise = (np.random.rand(h_len) - 0.5) * np.exp(-70 * h_t)
            audio[i:i+h_len] += hat_noise * 0.18

    # Final Master Mix & Normalization
    audio = audio / (np.max(np.abs(audio)) + 1e-5)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    byte_io = io.BytesIO()
    wav.write(byte_io, sample_rate, audio_int16)
    return byte_io.getvalue()

if st.button("🚀 Generate My Own Music", use_container_width=True):
    with st.spinner(f"⚡ Composing Unique {genre} ({mood}) Beat... Please wait..."):
        audio_bytes = generate_infinite_track(genre, mood)
        
        st.success(f"🎉 100% Unique Beat Generated! ({genre} - {mood})")
        st.audio(audio_bytes, format="audio/wav")
        st.download_button(
            label="⬇️ Download Track (.wav)",
            data=audio_bytes,
            file_name=f"SelfBeats_{genre.replace(' ', '_')}_{platform.replace(' ', '')}.wav",
            mime="audio/wav",
            use_container_width=True
        )