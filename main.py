import io
import time
import random
import numpy as np
import scipy.io.wavfile as wav
import streamlit as st

st.set_page_config(
    page_title="SelfBeats AI - Custom Instrument Studio",
    page_icon="🎼",
    layout="centered",
)

st.title("🎼 SelfBeats AI - Custom Studio")
st.subheader("Generate Solo Instruments or Mix 2 to 5 Instruments Together")

# 1. Selection Mode
generation_mode = st.radio(
    "Instrument Mode",
    ["Solo Instrument (Single)", "Custom Multi-Instrument Mix"],
    horizontal=True,
)

selected_instruments = []

if generation_mode == "Solo Instrument (Single)":
    solo_inst = st.selectbox(
        "Select Your Instrument",
        [
            "Acoustic Piano",
            "Synth Saw Lead",
            "808 Sub Bass",
            "Orchestral Brass Horns",
            "Ambient Synth Pad",
            "Electric Pluck",
        ],
    )
    selected_instruments.append(solo_inst)

else:
    st.markdown("**Select Instruments to Mix Together (Choose 1 to 5):**")
    col_inst1, col_inst2 = st.columns(2)
    with col_inst1:
        if st.checkbox("🎹 Acoustic Piano", value=True):
            selected_instruments.append("Acoustic Piano")
        if st.checkbox("🎸 Electric Pluck", value=True):
            selected_instruments.append("Electric Pluck")
        if st.checkbox("🎺 Orchestral Brass", value=False):
            selected_instruments.append("Orchestral Brass Horns")
        if st.checkbox("🥁 Drum Beat & Percussion", value=True):
            selected_instruments.append("Drums & Percussion")
    with col_inst2:
        if st.checkbox("🎛️ Synth Saw Lead", value=False):
            selected_instruments.append("Synth Saw Lead")
        if st.checkbox("🔊 808 Sub Bass", value=True):
            selected_instruments.append("808 Sub Bass")
        if st.checkbox("🌌 Ambient Synth Pad", value=False):
            selected_instruments.append("Ambient Synth Pad")

st.write("---")

# 2. Vibe & Scale Settings
col1, col2 = st.columns(2)
with col1:
    genre = st.selectbox(
        "Music Style / Vibe",
        ["Lo-Fi / Chill", "Phonk / Dark", "Cinematic / Epic", "Upbeat Pop", "Synthwave"],
    )
with col2:
    scale_type = st.selectbox(
        "Key Scale",
        [
            "Minor (Emotional/Dark)",
            "Major (Happy/Bright)",
            "Pentatonic (Catchy)",
            "Dorian (Mysterious)",
        ],
    )


def get_scale_notes(scale_type):
    scales = {
        "Minor (Emotional/Dark)": [
            110.0,
            123.47,
            130.81,
            146.83,
            164.81,
            174.61,
            196.00,
            220.00,
            261.63,
        ],
        "Major (Happy/Bright)": [
            130.81,
            146.83,
            164.81,
            174.61,
            196.00,
            220.00,
            246.94,
            261.63,
            293.66,
        ],
        "Pentatonic (Catchy)": [
            110.0,
            130.81,
            146.83,
            164.81,
            196.00,
            220.00,
            261.63,
        ],
        "Dorian (Mysterious)": [
            110.0,
            123.47,
            130.81,
            146.83,
            164.81,
            185.00,
            196.00,
            220.00,
        ],
    }
    return scales.get(scale_type, scales["Minor (Emotional/Dark)"])


def generate_custom_instrument_track(instruments, genre, scale_type, duration=15):
    sample_rate = 44100

    # Microsecond unique seed
    seed = int(time.time() * 1000) ^ random.randint(1000, 999999)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    total_samples = len(t)
    master_audio = np.zeros(total_samples)

    bpm = 140 if "Phonk" in genre else 85 if "Lo-Fi" in genre else 115
    beat_sec = 60.0 / bpm
    scale = get_scale_notes(scale_type)

    # 1. Render Drum Layer (If Selected)
    if "Drums & Percussion" in instruments:
        kick_step = int(beat_sec * sample_rate)
        for i in range(0, total_samples, kick_step):
            k_len = min(int(0.2 * sample_rate), total_samples - i)
            k_t = np.linspace(0, 0.2, k_len, False)
            kick = (
                np.sin(2 * np.pi * (120 * np.exp(-30 * k_t) + 40) * k_t)
                * np.exp(-8 * k_t)
            )
            master_audio[i : i + k_len] += kick * 0.7

        snare_step = int(beat_sec * sample_rate)
        for i in range(snare_step, total_samples, snare_step * 2):
            s_len = min(int(0.15 * sample_rate), total_samples - i)
            s_t = np.linspace(0, 0.15, s_len, False)
            snare = (np.random.rand(s_len) - 0.5) * np.exp(-25 * s_t)
            master_audio[i : i + s_len] += snare * 0.35

    # 2. Render Bass Layer (If Selected)
    if "808 Sub Bass" in instruments:
        bass_step = int(beat_sec * sample_rate)
        for i in range(0, total_samples, bass_step):
            b_len = min(int(0.4 * sample_rate), total_samples - i)
            b_t = np.linspace(0, 0.4, b_len, False)
            note = scale[0] / 2  # Sub octave
            bass = np.sin(2 * np.pi * note * b_t) * np.exp(-3 * b_t)
            master_audio[i : i + b_len] += bass * 0.6

    # 3. Render Melodic & Harmonic Instruments (Piano, Synth, Brass, Pad, Pluck)
    step = int((beat_sec / 2) * sample_rate)
    for i in range(0, total_samples, step):
        if random.random() > 0.2:
            note_freq = random.choice(scale)
            m_len = min(
                int((beat_sec * random.choice([0.5, 1.0, 2.0])) * sample_rate),
                total_samples - i,
            )
            m_t = np.linspace(0, m_len / sample_rate, m_len, False)

            melodic_wave = np.zeros(m_len)

            if "Acoustic Piano" in instruments:
                melodic_wave += (
                    0.5
                    * np.sin(2 * np.pi * note_freq * m_t)
                    * np.exp(-3.5 * m_t)
                )
            if "Synth Saw Lead" in instruments:
                melodic_wave += 0.35 * (m_t * note_freq % 1 - 0.5)
            if "Electric Pluck" in instruments:
                melodic_wave += (
                    0.45
                    * np.sin(2 * np.pi * note_freq * m_t)
                    * np.exp(-12 * m_t)
                )
            if "Orchestral Brass Horns" in instruments:
                melodic_wave += 0.4 * np.sin(
                    2 * np.pi * note_freq * m_t
                ) + 0.2 * np.sin(2 * np.pi * (note_freq * 2) * m_t)
            if "Ambient Synth Pad" in instruments:
                melodic_wave += (
                    0.3
                    * np.sin(2 * np.pi * note_freq * m_t)
                    * (1 - np.exp(-2 * m_t))
                )

            master_audio[i : i + m_len] += melodic_wave * 0.4

    # Master Normalization
    master_audio = master_audio / (np.max(np.abs(master_audio)) + 1e-5)
    audio_int16 = (master_audio * 32767).astype(np.int16)

    byte_io = io.BytesIO()
    wav.write(byte_io, sample_rate, audio_int16)
    return byte_io.getvalue()


if st.button("🚀 Render Custom Track", use_container_width=True):
    if not selected_instruments:
        st.warning("⚠️ Please select at least one instrument!")
    else:
        with st.spinner(
            f"🎧 Synthesizing audio with: {', '.join(selected_instruments)}..."
        ):
            audio_data = generate_custom_instrument_track(
                selected_instruments, genre, scale_type
            )

            st.success(
                f"🎉 Created track using {len(selected_instruments)} selected instrument(s)!"
            )
            st.audio(audio_data, format="audio/wav")
            st.download_button(
                label="⬇️ Download Track (.wav)",
                data=audio_data,
                file_name="SelfBeats_Custom_Track.wav",
                mime="audio/wav",
                use_container_width=True,
            )