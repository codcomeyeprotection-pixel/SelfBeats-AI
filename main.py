import io, time, random, numpy as np, scipy.io.wavfile as wav, scipy.signal as signal, streamlit as st
from pydub import AudioSegment

st.set_page_config(page_title="SelfBeats AI", page_icon="logo.png", layout="wide")

STANDARD_SAMPLE_RATE = 44100
AUDIO_BUFFER_SIZE = 32768
INTERNAL_DTYPE = np.float64
MIN_ENVELOPE_TIME_SEC = 0.005
MASTER_HEADROOM_DB = -3.0
MASTER_HEADROOM_GAIN = np.float64(10 ** (MASTER_HEADROOM_DB / 20.0))
MASTER_LIMITER_DBFS = -0.5
MASTER_LIMITER_THRESHOLD = np.float64(10 ** (MASTER_LIMITER_DBFS / 20.0))
MASTER_TARGET_DBFS = -0.3
MASTER_TARGET_PEAK = np.float64(10 ** (MASTER_TARGET_DBFS / 20.0))

logo_col, title_col = st.columns([1, 8])
with logo_col:
    st.image("logo.png", width=100)
with title_col:
    st.title("SelfBeats AI - Ultimate Mega Studio")

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


def lowpass_filter(data, cutoff=3200, fs=STANDARD_SAMPLE_RATE):
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.98)
    b, a = signal.butter(4, normal_cutoff, btype="low", analog=False)
    return np.asarray(
        signal.lfilter(b, a, np.asarray(data, dtype=INTERNAL_DTYPE)),
        dtype=INTERNAL_DTYPE,
    )


def band_limit_filter(data, fs=STANDARD_SAMPLE_RATE):
    """Remove subsonic and harsh digital frequencies on every audio path."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    if len(data) < 8:
        return data
    highpass = signal.butter(6, 20.0, btype="highpass", fs=fs, output="sos")
    lowpass = signal.butter(6, 18000.0, btype="lowpass", fs=fs, output="sos")
    filtered = signal.sosfilt(highpass, data)
    filtered = signal.sosfilt(lowpass, filtered)
    return np.asarray(filtered, dtype=INTERNAL_DTYPE)


def body_reverb(data, delay_ms=35, decay=0.28, fs=STANDARD_SAMPLE_RATE):
    delay_samples = max(1, int((delay_ms / 1000.0) * fs))
    output = np.array(data, dtype=INTERNAL_DTYPE, copy=True)
    for buffer_start in range(0, len(data), AUDIO_BUFFER_SIZE):
        buffer_end = min(buffer_start + AUDIO_BUFFER_SIZE, len(data))
        first_sample = max(delay_samples, buffer_start)
        for i in range(first_sample, buffer_end):
            output[i] += INTERNAL_DTYPE(output[i - delay_samples] * decay)
    return output


def smooth_adsr(
    n_samples,
    fs=STANDARD_SAMPLE_RATE,
    attack_sec=0.04,
    decay_sec=0.08,
    sustain_level=0.82,
    release_sec=0.14,
):
    """Create a click-safe ADSR envelope with eased attack and release."""
    if n_samples <= 0:
        return np.zeros(0, dtype=INTERNAL_DTYPE)

    attack_sec = max(MIN_ENVELOPE_TIME_SEC, attack_sec)
    release_sec = max(MIN_ENVELOPE_TIME_SEC, release_sec)
    envelope = np.ones(n_samples, dtype=INTERNAL_DTYPE) * INTERNAL_DTYPE(sustain_level)
    attack_samples = min(max(1, int(attack_sec * fs)), max(1, n_samples // 3))
    release_samples = min(
        max(1, int(release_sec * fs)),
        max(1, (n_samples - attack_samples) // 2),
    )
    decay_samples = min(
        max(0, int(decay_sec * fs)),
        max(0, n_samples - attack_samples - release_samples),
    )

    if attack_samples:
        attack_phase = np.linspace(0.0, np.pi / 2, attack_samples, dtype=INTERNAL_DTYPE)
        envelope[:attack_samples] = np.sin(attack_phase) ** 2

    decay_start = attack_samples
    decay_end = decay_start + decay_samples
    if decay_samples:
        decay_phase = np.linspace(0.0, np.pi / 2, decay_samples, dtype=INTERNAL_DTYPE)
        envelope[decay_start:decay_end] = (
            1.0 - (1.0 - sustain_level) * np.sin(decay_phase) ** 2
        )

    if release_samples:
        release_start = n_samples - release_samples
        release_phase = np.linspace(0.0, np.pi / 2, release_samples, dtype=INTERNAL_DTYPE)
        envelope[release_start:] = INTERNAL_DTYPE(sustain_level) * np.cos(release_phase) ** 2

    envelope[0] = 0.0
    envelope[-1] = 0.0
    return envelope


def soft_limiter(data, threshold=MASTER_LIMITER_THRESHOLD, drive=1.25):
    """Apply a soft knee followed by a brickwall ceiling."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    threshold = INTERNAL_DTYPE(threshold)
    driven = data * INTERNAL_DTYPE(drive)
    limited = threshold * np.tanh(driven / threshold)
    limited /= np.tanh(INTERNAL_DTYPE(drive))
    return np.clip(limited, -threshold, threshold).astype(INTERNAL_DTYPE)


def normalize_master(data, target_peak=MASTER_TARGET_PEAK):
    """Peak-normalize the float64 master to the requested dBFS headroom."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    peak = np.max(np.abs(data))
    if peak <= 0:
        return data
    normalized = data * (INTERNAL_DTYPE(target_peak) / INTERNAL_DTYPE(peak))
    return np.clip(normalized, -1.0, 1.0).astype(INTERNAL_DTYPE)


def wav_to_mp3(wav_bytes, bitrate="320k"):
    """Convert the rendered WAV bytes to a widely compatible MP3 download."""
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        mp3_io = io.BytesIO()
        audio_segment.export(
            mp3_io,
            format="mp3",
            bitrate=bitrate,
            parameters=["-ar", str(STANDARD_SAMPLE_RATE)],
        )
    except Exception as exc:
        raise RuntimeError("MP3 conversion is unavailable in this environment.") from exc
    return mp3_io.getvalue()


def _synthesize_raw_sound(inst, freq, length_sec, fs=STANDARD_SAMPLE_RATE):
    n_samples = max(1, int(length_sec * fs))
    t = np.arange(n_samples, dtype=INTERNAL_DTYPE) / INTERNAL_DTYPE(fs)

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
        vibrato = 1.0 + INTERNAL_DTYPE(0.009) * np.sin(2 * np.pi * 5.5 * t)
        breath = (np.random.rand(n_samples).astype(INTERNAL_DTYPE) - 0.5) * INTERNAL_DTYPE(0.06)
        phase = np.cumsum(INTERNAL_DTYPE(freq) * vibrato) / INTERNAL_DTYPE(fs)
        core = np.sin(2 * np.pi * phase) + INTERNAL_DTYPE(0.3) * np.sin(
            2 * np.pi * 2 * phase
        )
        env = smooth_adsr(n_samples, fs, attack_sec=0.035, decay_sec=0.09, sustain_level=0.76, release_sec=0.16)
        return lowpass_filter((core + breath) * env, cutoff=4200, fs=fs)

    elif any(p in inst for p in ["Tabla", "Dholak", "Cajón", "Bongos", "Congas"]):
        pitch_drop = INTERNAL_DTYPE(freq) * np.exp(-32 * t) + (INTERNAL_DTYPE(freq) * 0.35)
        base = np.sin(2 * np.pi * pitch_drop * t) * np.exp(-10 * t)
        slap = (np.random.rand(n_samples).astype(INTERNAL_DTYPE) - 0.5) * np.exp(-55 * t) * 0.25
        return lowpass_filter(base + slap, cutoff=1900, fs=fs)

    elif any(g in inst for g in ["Guitar", "Ukulele", "Sitar"]):
        period = int(fs / max(freq, 50))
        buf = np.random.uniform(-1, 1, period).astype(INTERNAL_DTYPE)
        sound = np.zeros(n_samples, dtype=INTERNAL_DTYPE)
        for i in range(n_samples):
            sound[i] = buf[0]
            avg = INTERNAL_DTYPE(0.5 * (buf[0] + buf[1]) * 0.982)
            buf = np.append(buf[1:], avg)
        return sound

    elif any(s in inst for s in ["Violin", "Viola", "Cello", "Double Bass"]):
        vibrato = 1.0 + INTERNAL_DTYPE(0.012) * np.sin(2 * np.pi * 6 * t)
        phase = np.cumsum(INTERNAL_DTYPE(freq) * vibrato) / INTERNAL_DTYPE(fs)
        saw = INTERNAL_DTYPE(2.0) * (phase % INTERNAL_DTYPE(1.0)) - INTERNAL_DTYPE(1.0)
        bow_env = smooth_adsr(n_samples, fs, attack_sec=0.055, decay_sec=0.12, sustain_level=0.84, release_sec=0.2)
        return lowpass_filter(saw * bow_env, cutoff=2900, fs=fs)

    elif any(k in inst for k in ["Piano", "Keyboard"]):
        harmonics = (
            1.0 * np.sin(2 * np.pi * freq * t) * np.exp(-2.8 * t)
            + 0.45 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-4.5 * t)
            + 0.2 * np.sin(2 * np.pi * freq * 3 * t) * np.exp(-7.0 * t)
        )
        return np.asarray(harmonics, dtype=INTERNAL_DTYPE)

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
            ).astype(INTERNAL_DTYPE) * np.exp(-7 * t)
        else:
            decay = (
                75
                if any(
                    x in inst
                    for x in ["Hi-Hat", "Shaker", "Tambourine", "Triangle", "Maracas"]
                )
                else 25
            )
            return (np.random.rand(n_samples).astype(INTERNAL_DTYPE) - 0.5) * np.exp(-decay * t)

    else:
        return np.asarray(
            np.sin(2 * np.pi * freq * t) * np.exp(-3.5 * t),
            dtype=INTERNAL_DTYPE,
        )


def synthesize_hifi_sound(inst, freq, length_sec, fs=STANDARD_SAMPLE_RATE):
    """Render and protect every instrument, controller, and gear sound path."""
    raw_sound = _synthesize_raw_sound(inst, freq, length_sec, fs)
    if len(raw_sound) == 0:
        return np.zeros(0, dtype=INTERNAL_DTYPE)

    is_percussive = any(
        marker in inst
        for marker in [
            "Drum", "Kick", "Snare", "Tom", "Cymbal", "Shaker",
            "Tambourine", "Triangle", "Maracas", "Tabla", "Dholak",
            "Bongos", "Congas",
        ]
    )
    attack_sec = 0.008 if is_percussive else 0.02
    release_sec = 0.012 if is_percussive else 0.04
    envelope = smooth_adsr(
        len(raw_sound),
        fs=fs,
        attack_sec=max(MIN_ENVELOPE_TIME_SEC, attack_sec),
        decay_sec=0.04 if is_percussive else 0.08,
        sustain_level=1.0,
        release_sec=max(MIN_ENVELOPE_TIME_SEC, release_sec),
    )
    processed = np.asarray(raw_sound, dtype=INTERNAL_DTYPE) * envelope
    processed = band_limit_filter(processed, fs=fs)
    edge_fade = smooth_adsr(
        len(processed),
        fs=fs,
        attack_sec=MIN_ENVELOPE_TIME_SEC,
        decay_sec=0.0,
        sustain_level=1.0,
        release_sec=MIN_ENVELOPE_TIME_SEC,
    )
    return processed * edge_fade


def generate_track(inst_list, fx_list, is_auto=False, duration=15):
    fs = STANDARD_SAMPLE_RATE
    seed = int(time.time() * 1000) ^ random.randint(1000, 999999)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    total_samples = int(duration * fs)
    master = np.zeros(total_samples, dtype=INTERNAL_DTYPE)

    scale = [130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94, 261.63]
    bpm = random.randint(88, 128)
    beat_sec = 60.0 / bpm

    if is_auto:
        inst_list = random.sample(pure_instruments, k=random.randint(3, 6))

    for inst in inst_list:
        layer = np.zeros(total_samples, dtype=INTERNAL_DTYPE)
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
                layer[i : i + avail] += sound[:avail] * INTERNAL_DTYPE(0.35)

        master += layer

    # Gain-stage the complete mix before any nonlinear processing.
    master *= MASTER_HEADROOM_GAIN

    if any(
        f in fx_list
        for f in ["Distortion Pedal", "Overdrive Pedal", "Fuzz Pedal", "Saturation Unit"]
    ):
        master = soft_limiter(master * INTERNAL_DTYPE(1.7), threshold=0.75, drive=1.1)

    master = body_reverb(master, fs=fs)
    master = band_limit_filter(master, fs=fs)
    master = soft_limiter(master, threshold=MASTER_LIMITER_THRESHOLD, drive=1.1)
    master = normalize_master(master)
    audio_int16 = np.rint(master * INTERNAL_DTYPE(32767)).astype(np.int16)

    byte_io = io.BytesIO()
    wav.write(byte_io, fs, audio_int16)
    return byte_io.getvalue(), inst_list


if st.button("🚀 Render Music Track", use_container_width=True):
    is_auto = mode == "⚡ 1-Click Auto"
    if not is_auto and not selected_instruments:
        st.warning("⚠️ Please select at least one instrument.")
    else:
        with st.spinner("🎧 Synthesizing DSP Audio..."):
            audio_wav, used = generate_track(
                selected_instruments, selected_fx, is_auto=is_auto
            )
            st.success("🎉 Composition Generated Successfully!")
            st.markdown(f"**Active Rendered Instruments:** {', '.join(set(used))}")
            st.audio(audio_wav, format="audio/wav")

            audio_mp3 = None
            try:
                audio_mp3 = wav_to_mp3(audio_wav)
            except RuntimeError:
                st.warning("MP3 export is unavailable right now, but the WAV download is still ready.")

            download_cols = st.columns(2)
            with download_cols[0]:
                st.download_button(
                    "⬇️ Download WAV",
                    data=audio_wav,
                    file_name="SelfBeats_Studio_Track.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )
            with download_cols[1]:
                if audio_mp3 is not None:
                    st.download_button(
                        "⬇️ Download MP3",
                        data=audio_mp3,
                        file_name="SelfBeats_Studio_Track.mp3",
                        mime="audio/mpeg",
                        use_container_width=True,
                    )