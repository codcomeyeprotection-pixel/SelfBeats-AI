import io, time, random, numpy as np, scipy.io.wavfile as wav, scipy.signal as signal, streamlit as st

st.set_page_config(page_title="SelfBeats AI Mega DAW", page_icon="🎛️", layout="wide")
st.title("🎛️ SelfBeats AI - Mega Studio DAW")

mode = st.radio("Select Workflow:", ["⚡ 1-Click Auto", "🎼 Pure Instruments Only", "🎛️ Full Hardware Rack (Step-by-Step)"], horizontal=True)

pure_instruments = [
    "Acoustic Piano", "Digital Piano", "Electric Piano", "Keyboard", "Synthesizer", "Sampler",
    "Drum Machine", "Drum Kit", "Electronic Drum Kit", "Bass Guitar", "Electric Guitar", "Acoustic Guitar",
    "Classical Guitar", "Ukulele", "Violin", "Viola", "Cello", "Double Bass", "Flute", "Saxophone",
    "Clarinet", "Trumpet", "Trombone", "Harmonica", "Accordion", "Tabla", "Dholak", "Cajón",
    "Bongos", "Congas", "Tambourine", "Shaker", "Triangle", "Maracas"
]

midi_controllers = ["MIDI Keyboard", "MIDI Pad Controller", "MIDI Drum Pad", "MIDI Fader", "MIDI Foot Ctrl", "MIDI Guitar Ctrl", "MIDI Wind Ctrl", "Launchpad", "Groovebox", "Sequencer", "Control Surface"]
mixing_equipment = ["Mixing Console", "Digital Mixer", "Analog Mixer", "Fader Bank", "Channel Strip", "Compressor", "Limiter", "Equalizer", "Gate", "Expander", "Reverb Processor", "Delay Processor", "Multi-FX"]
synth_electronic = ["Analog Synth", "Digital Synth", "Modular Synth", "Eurorack", "Bass Synth", "Vocoder", "Arpeggiator", "Step Sequencer", "Effects Pedals"]
guitar_bass_gear = ["Guitar Amp", "Bass Amp", "Amp Head", "Speaker Cabinet", "Distortion Pedal", "Overdrive Pedal", "Fuzz Pedal", "Chorus Pedal", "Delay Pedal", "Reverb Pedal", "Compressor Pedal", "Wah Pedal", "Tuner", "DI Box"]
drum_equipment = ["Kick Drum", "Snare Drum", "Tom", "Floor Tom", "Hi-Hat", "Crash Cymbal", "Ride Cymbal", "Splash Cymbal", "China Cymbal", "Drum Throne", "Sticks", "Brushes", "Kick Pedal", "Drum Mic Set"]
mastering_equipment = ["Mastering Compressor", "Mastering EQ", "Stereo Imager", "Limiter", "Saturation Unit", "Tape Machine", "Analog Console", "Reference DAC", "Loudness Meter"]

selected_instruments, selected_fx = [], []

if mode == "🎼 Pure Instruments Only":
    st.markdown("### 🎼 Choose Instruments:")
    cols = st.columns(3)
    for idx, inst in enumerate(pure_instruments):
        if cols[idx % 3].checkbox(inst, value=(inst in ["Acoustic Piano", "Tabla", "Flute"]), key=f"pure_{inst}"):
            selected_instruments.append(inst)

elif mode == "🎛️ Full Hardware Rack (Step-by-Step)":
    st.markdown("### 🎛️ Custom Studio Rack Configuration")
    categories = [
        ("🎹 Primary Instruments", pure_instruments, True, selected_instruments),
        ("🎛️ MIDI & Controllers", midi_controllers, False, None),
        ("🎚️ Mixing Equipment", mixing_equipment, False, selected_fx),
        ("⚡ Synths & Electronic", synth_electronic, False, selected_instruments),
        ("🎸 Guitar & Bass Gear", guitar_bass_gear, False, selected_fx),
        ("🥁 Drum Rig Equipment", drum_equipment, False, selected_instruments),
        ("🎛️ Mastering Chain", mastering_equipment, False, selected_fx),
    ]
    for label, items, expand, target_list in categories:
        with st.expander(label, expanded=expand):
            cols = st.columns(3)
            for idx, item in enumerate(items):
                if cols[idx % 3].checkbox(item, key=f"rack_{label}_{item}"):
                    if target_list is not None:
                        target_list.append(item)


def lowpass_filter(data, cutoff=3200, fs=44100):
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.98)
    b, a = signal.butter(2, normal_cutoff, btype="low", analog=False)
    return signal.lfilter(b, a, data)


def body_reverb(data, delay_ms=35, decay=0.28):
    delay_samples = int((delay_ms / 1000.0) * 44100)
    output = np.copy(data)
    for i in range(delay_samples, len(data)):
        output[i] += output[i - delay_samples] * decay
    return output


def synthesize_hifi_sound(inst, freq, length_sec, fs=44100):
    n_samples = int(length_sec * fs)
    t = np.linspace(0, length_sec, n_samples, False)

    if any(
        w in inst
        for w in [
            "Flute",
            "Harmonica",
            "Saxophone",
            "Clarinet",
            "Trumpet",
            "Trombone",
            "Accordion",
        ]
    ):
        vibrato = 1.0 + 0.009 * np.sin(2 * np.pi * 5.5 * t)
        breath = (np.random.rand(n_samples) - 0.5) * 0.06
        core = np.sin(2 * np.pi * freq * vibrato * t) + 0.3 * np.sin(
            2 * np.pi * (freq * 2) * t
        )
        env = (1 - np.exp(-12 * t)) * np.exp(-2.2 * t)
        return lowpass_filter((core + breath) * env, cutoff=4200)

    elif any(p in inst for p in ["Tabla", "Dholak", "Cajón", "Bongos", "Congas"]):
        pitch_drop = freq * np.exp(-32 * t) + (freq * 0.35)
        base = np.sin(2 * np.pi * pitch_drop * t) * np.exp(-10 * t)
        slap = (np.random.rand(n_samples) - 0.5) * np.exp(-55 * t) * 0.25
        return lowpass_filter(base + slap, cutoff=1900)

    elif any(g in inst for g in ["Guitar", "Ukulele", "Sitar"]):
        period = int(fs / max(freq, 50))
        buf = np.random.uniform(-1, 1, period)
        sound = np.zeros(n_samples)
        for i in range(n_samples):
            sound[i] = buf[0]
            avg = 0.5 * (buf[0] + buf[1]) * 0.982
            buf = np.append(buf[1:], avg)
        return sound

    elif any(s in inst for s in ["Violin", "Viola", "Cello", "Double Bass"]):
        vibrato = 1.0 + 0.012 * np.sin(2 * np.pi * 6 * t)
        saw = 2 * (t * freq * vibrato % 1) - 1
        bow_env = (1 - np.exp(-4 * t)) * np.exp(-1.2 * t)
        return lowpass_filter(saw * bow_env, cutoff=2900)

    elif any(k in inst for k in ["Piano", "Keyboard"]):
        harmonics = (
            1.0 * np.sin(2 * np.pi * freq * t) * np.exp(-2.8 * t)
            + 0.45 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-4.5 * t)
            + 0.2 * np.sin(2 * np.pi * freq * 3 * t) * np.exp(-7.0 * t)
        )
        return harmonics

    elif any(
        d in inst
        for d in [
            "Kick",
            "Snare",
            "Tom",
            "Hi-Hat",
            "Cymbal",
            "Drum",
            "Shaker",
            "Tambourine",
            "Triangle",
            "Maracas",
        ]
    ):
        if "Kick" in inst:
            return np.sin(
                2 * np.pi * (140 * np.exp(-38 * t) + 38) * t
            ) * np.exp(-7 * t)
        else:
            decay = (
                75
                if any(
                    x in inst
                    for x in ["Hi-Hat", "Shaker", "Tambourine", "Triangle", "Maracas"]
                )
                else 25
            )
            return (np.random.rand(n_samples) - 0.5) * np.exp(-decay * t)

    else:
        return np.sin(2 * np.pi * freq * t) * np.exp(-3.5 * t)


def generate_track(inst_list, fx_list, is_auto=False, duration=15):
    fs = 44100
    seed = int(time.time() * 1000) ^ random.randint(1000, 999999)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    total_samples = int(duration * fs)
    master = np.zeros(total_samples)

    scale = [130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94, 261.63]
    bpm = random.randint(88, 128)
    beat_sec = 60.0 / bpm

    if is_auto:
        inst_list = random.sample(pure_instruments, k=random.randint(3, 6))

    for inst in inst_list:
        layer = np.zeros(total_samples)
        step = int(
            (
                beat_sec
                / (4 if any(p in inst for p in ["Hi-Hat", "Tabla", "Shaker", "Drum"]) else 2)
            )
            * fs
        )

        for i in range(0, total_samples, step):
            if random.random() > 0.2:
                freq = random.choice(scale)
                n_len = random.choice([0.4, 0.8, 1.2])
                sound = synthesize_hifi_sound(inst, freq, n_len, fs)

                avail = min(len(sound), total_samples - i)
                layer[i : i + avail] += sound[:avail] * 0.35

        master += layer

    if any(
        f in fx_list
        for f in ["Distortion Pedal", "Overdrive Pedal", "Fuzz Pedal", "Saturation Unit"]
    ):
        master = np.clip(master * 1.7, -0.75, 0.75)

    master = body_reverb(master)
    master = master / (np.max(np.abs(master)) + 1e-5)
    audio_int16 = (master * 32767).astype(np.int16)

    byte_io = io.BytesIO()
    wav.write(byte_io, fs, audio_int16)
    return byte_io.getvalue(), inst_list


if st.button("🚀 Render Music Track", use_container_width=True):
    is_auto = mode == "⚡ 1-Click Auto"
    if not is_auto and not selected_instruments:
        st.warning("⚠️ Please select at least one instrument.")
    else:
        with st.spinner("🎧 Synthesizing DSP Audio..."):
            audio, used = generate_track(
                selected_instruments, selected_fx, is_auto=is_auto
            )
            st.success("🎉 Composition Generated Successfully!")
            st.markdown(f"**Active Rendered Instruments:** {', '.join(set(used))}")
            st.audio(audio, format="audio/wav")
            st.download_button(
                "⬇️ Download High-Quality WAV",
                data=audio,
                file_name="SelfBeats_Studio_Track.wav",
                mime="audio/wav",
                use_container_width=True,
            )