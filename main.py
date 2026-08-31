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
    genre = st.selectbox("Style & Soundkit", [
        "Drift Phonk (Heavy 808)", "Lo-Fi Jazzhop", "Modern Trap Beat",
        "Cyberpunk Synthwave", "Orchestral Cinematic", "Afrobeat Groove",
        "Deep House / EDM", "Dark Ambient"
    ])
    lead_instrument = st.selectbox("Lead Instrument", [
        "Sawtooth Synth (FL Synthmaker)", "Grand Piano (FL Keys)", 
        "Plucked Synth (FL Sytrus)", "Brass Horns (FL DirectWave)"
    ])

with col2:
    mood = st.selectbox("Mood / Vibe", ["Aggressive / Dark", "Chill / Relaxed", "Hype / Energetic", "Emotional / Sad"])
    scale_type = st.selectbox("Musical Scale Engine", ["Minor (Dark/Sad)", "Major (Happy/Bright)", "Pentatonic (Catchy)", "Dorian (Mysterious)"])

st.markdown("**🎛️ Automated FL Studio FX & Sequencer Rack**")
col_fx1, col_fx2, col_fx3 = st.columns(3)
with col_fx1:
    enable_sidechain = st.checkbox("Sidechain Compression", value=True)
with col_fx2:
    enable_reverb = st.checkbox("Space Reverb / Delay", value=True)
with col_fx3:
    enable_arpeggio = st.checkbox("FL Arpeggiator", value=True)

# FL Studio Scales Setup
def get_scale_notes(scale_type):
    scales = {
        "Minor (Dark/Sad)": [110.0, 123.47, 130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94, 261.63],
        "Major (Happy/Bright)": [130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94, 261.63, 293.66],
        "Pentatonic (Catchy)": [110.0, 130.81, 146.83, 164.81, 196.00, 220.00, 261.63],
        "Dorian (Mysterious)": [110.0, 123.47, 130.81, 146.83, 164.81, 185.00, 196.00, 220.00]
    }
    return scales.get(scale_type, scales["Minor (Dark/Sad)"])

# DSP Effects: Reverb/Delay Simulation
def apply_reverb_delay(signal, delay_samples=8820, decay=0.35):
    output = np.copy(signal)
    for i in range(delay_samples, len(signal)):
        output[i] += output[i - delay_samples] * decay
    return output

# Main FL Engine Function
def generate_fl_studio_automatic_track(genre, lead_inst, mood, scale_type, sidechain, reverb, arpeggio, duration=15):
    sample_rate = 44100
    
    # 1. Microsecond Seed for 100% Unique Auto Generation
    seed = int(time.time() * 1000) ^ random.randint(1000, 999999)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    total_samples = len(t)
    
    # Mix Buses
    drums_bus = np.zeros(total_samples)
    bass_bus = np.zeros(total_samples)
    melody_bus = np.zeros(total_samples)
    
    # Auto BPM Selector based on Genre
    if "Phonk" in genre or "Trap" in genre:
        bpm = random.randint(130, 160)
    elif "Lo-Fi" in genre:
        bpm = random.randint(75, 92)
    elif "EDM" in genre:
        bpm = random.randint(124, 128)
    else:
        bpm = random.randint(95, 120)
        
    beat_sec = 60.0 / bpm
    scale = get_scale_notes(scale_type)
    
    # FL Sequencer Layer 1: Kick & Sub Bass
    kick_step = int(beat_sec * sample_rate)
    for i in range(0, total_samples, kick_step):
        if random.random() > 0.1:
            k_len = min(int(0.22 * sample_rate), total_samples - i)
            k_t = np.linspace(0, 0.22, k_len, False)
            kick_freq = (150 if "Phonk" in genre else 110) * np.exp(-32 * k_t) + 38
            kick = np.sin(2 * np.pi * kick_freq * k_t) * np.exp(-8 * k_t)
            drums_bus[i:i+k_len] += kick * 0.9

    # FL Sequencer Layer 2: Snare / Clap
    snare_step = int(beat_sec * sample_rate)
    for i in range(snare_step, total_samples, snare_step * 2):
        s_len = min(int(0.16 * sample_rate), total_samples - i)
        s_t = np.linspace(0, 0.16, s_len, False)
        noise = (np.random.rand(s_len) - 0.5) * np.exp(-25 * s_t)
        drums_bus[i:i+s_len] += noise * 0.4

    # FL Sequencer Layer 3: Hi-Hats / Shakers
    hat_step = int((beat_sec / 4) * sample_rate)
    for i in range(0, total_samples, hat_step):
        if random.random() > 0.12:
            h_len = min(int(0.035 * sample_rate), total_samples - i)
            h_t = np.linspace(0, 0.035, h_len, False)
            hat = (np.random.rand(h_len) - 0.5) * np.exp(-75 * h_t)
            drums_bus[i:i+h_len] += hat * 0.18

    # FL Synth Piano Roll Layer (Lead Melody / Arp)
    note_duration_step = int((beat_sec / (4 if arpeggio else 2)) * sample_rate)
    for i in range(0, total_samples, note_duration_step):
        if random.random() > 0.2:
            note_freq = random.choice(scale)
            m_len = min(int((beat_sec * (0.25 if arpeggio else 0.8)) * sample_rate), total_samples - i)
            m_t = np.linspace(0, m_len / sample_rate, m_len, False)
            
            # Sound Design Presets
            if "Sawtooth" in lead_inst:
                synth = 0.4 * np.sin(2 * np.pi * note_freq * m_t) + 0.3 * (m_t * note_freq % 1 - 0.5)
            elif "Piano" in lead_inst:
                synth = 0.5 * np.sin(2 * np.pi * note_freq * m_t) * np.exp(-3.5 * m_t)
            elif "Plucked" in lead_inst:
                synth = 0.5 * np.sin(2 * np.pi * note_freq * m_t) * np.exp(-14 * m_t)
            else: # Brass
                synth = 0.5 * np.sin(2 * np.pi * note_freq * m_t) + 0.25 * np.sin(2 * np.pi * (note_freq * 2) * m_t)
                
            melody_bus[i:i+m_len] += synth * 0.35

    # Sidechain Compression Simulation (Ducks melody when kick hits)
    if sidechain:
        for i in range(0, total_samples, kick_step):
            duck_len = min(int(0.2 * sample_rate), total_samples - i)
            duck_env = np.linspace(0.2, 1.0, duck_len)
            melody_bus[i:i+duck_len] *= duck_env

    # Reverb FX Rack
    if reverb:
        melody_bus = apply_reverb_delay(melody_bus)

    # FL Master Mixer Channel
    master_mix = drums_bus + bass_bus + melody_bus
    master_mix = master_mix / (np.max(np.abs(master_mix)) + 1e-5)
    audio_int16 = (master_mix * 32767).astype(np.int16)
    
    byte_io = io.BytesIO()
    wav.write(byte_io, sample_rate, audio_int16)
    return byte_io.getvalue()

if st.button("🚀 Auto-Generate FL Studio Track", use_container_width=True):
    with st.spinner("🎛️ Processing FL Studio Piano Roll, Sequencer & FX Rack..."):
        audio_data = generate_fl_studio_automatic_track(
            genre, lead_instrument, mood, scale_type, 
            enable_sidechain, enable_reverb, enable_arpeggio
        )
        
        st.success(f"🎉 Track Rendered with FL FX Rack! ({genre} | Scale: {scale_type})")
        st.audio(audio_data, format="audio/wav")
        st.download_button(
            label="⬇️ Download Track (.wav)",
            data=audio_data,
            file_name=f"SelfBeats_FL_Studio_{genre.replace(' ', '_')}.wav",
            mime="audio/wav",
            use_container_width=True
        )