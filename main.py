import io
import random
import subprocess
import time

import imageio_ffmpeg
import numpy as np
import scipy.signal as signal
import soundfile as sf
import streamlit as st

st.set_page_config(page_title="SelfBeats AI", page_icon="logo.png", layout="wide")

STANDARD_SAMPLE_RATE = 44100
AUDIO_BUFFER_SIZE = 32768
INTERNAL_DTYPE = np.float64
MIN_ENVELOPE_TIME_SEC = 0.005
MASTER_HEADROOM_DB = -3.0
MASTER_HEADROOM_GAIN = np.float64(10 ** (MASTER_HEADROOM_DB / 20.0))
MASTER_GAIN_DB = 4.0
MASTER_GAIN = np.float64(10 ** (MASTER_GAIN_DB / 20.0))
MASTER_LIMITER_DBFS = -0.3
MASTER_LIMITER_THRESHOLD = np.float64(10 ** (MASTER_LIMITER_DBFS / 20.0))
MASTER_OUTPUT_DBFS = -0.1
MASTER_OUTPUT_PEAK = np.float64(10 ** (MASTER_OUTPUT_DBFS / 20.0))
PRESENCE_BOOST_DB = 1.5

BEAT_STYLES = [
    "Hip-Hop / Trap",
    "Modern Drill",
    "Boom Bap",
    "Lofi Beats",
    "Modern Pop / R&B",
]
SCALE_INTERVALS = {
    "Major": (0, 2, 4, 5, 7, 9, 11),
    "Minor": (0, 2, 3, 5, 7, 8, 10),
    "Harmonic Minor": (0, 2, 3, 5, 7, 8, 11),
    "Pentatonic": (0, 3, 5, 7, 10),
    "Dorian": (0, 2, 3, 5, 7, 9, 10),
    "Phrygian": (0, 1, 3, 5, 7, 8, 10),
}
CHORD_PROGRESSIONS = {
    "Major": (
        ("I–V–vi–IV", (0, 4, 5, 3)),
        ("ii–V–I–vi", (1, 4, 0, 5)),
        ("vi–IV–I–V", (5, 3, 0, 4)),
    ),
    "Minor": (
        ("i–VI–III–VII", (0, 5, 2, 6)),
        ("iiø–V–i–VI", (1, 4, 0, 5)),
        ("i–iv–VII–III", (0, 3, 6, 2)),
    ),
}
MELODY_MOTIFS = (
    (0, 2, 4, 2, 1, 2, 4, 5),
    (0, 0, 2, 4, 2, 1, 0, 2),
    (4, 2, 1, 0, 2, 4, 5, 4),
    (0, 2, 1, 4, 2, 0, 1, 2),
)
RESPONSE_MOTIFS = (
    (4, 2, 0, 2),
    (2, 1, 0, 1),
    (5, 4, 2, 0),
    (0, 2, 4, 2),
)


def _step_pattern(*steps):
    pattern = np.zeros(16, dtype=np.int8)
    pattern[list(steps)] = 1
    return pattern


BEAT_STYLE_PROFILES = {
    "Hip-Hop / Trap": {
        "bpm_range": (120, 160),
        "swing": 0.08,
        "scales": ("Minor", "Harmonic Minor", "Pentatonic", "Phrygian"),
        "melody_instruments": ("Bell Pluck", "Lead Synth"),
        "patterns": (
            {
                "kick": _step_pattern(0, 3, 7, 10, 12),
                "snare": _step_pattern(4, 12),
                "hat": _step_pattern(0, 2, 4, 6, 8, 10, 12, 14),
                "open_hat": _step_pattern(7, 15),
                "808": _step_pattern(0, 3, 7, 10, 12),
            },
            {
                "kick": _step_pattern(0, 5, 8, 11, 14),
                "snare": _step_pattern(4, 12),
                "hat": _step_pattern(0, 2, 4, 6, 8, 10, 12, 14),
                "open_hat": _step_pattern(3, 11),
                "808": _step_pattern(0, 5, 8, 11, 14),
            },
        ),
    },
    "Modern Drill": {
        "bpm_range": (135, 155),
        "swing": 0.16,
        "scales": ("Minor", "Harmonic Minor", "Phrygian", "Dorian"),
        "melody_instruments": ("Pluck Synth", "Bell Pluck", "Lead Synth"),
        "patterns": (
            {
                "kick": _step_pattern(0, 6, 7, 10, 14),
                "snare": _step_pattern(4, 12),
                "hat": _step_pattern(0, 2, 3, 4, 6, 8, 10, 11, 12, 14),
                "open_hat": _step_pattern(7, 15),
                "808": _step_pattern(0, 6, 7, 10, 14),
            },
            {
                "kick": _step_pattern(0, 3, 6, 9, 13),
                "snare": _step_pattern(4, 12),
                "hat": _step_pattern(0, 2, 4, 6, 8, 10, 12, 14),
                "open_hat": _step_pattern(3, 11),
                "808": _step_pattern(0, 3, 6, 9, 13),
            },
        ),
    },
    "Boom Bap": {
        "bpm_range": (82, 102),
        "swing": 0.2,
        "scales": ("Minor", "Major", "Dorian", "Pentatonic"),
        "melody_instruments": ("Piano", "Electric Piano", "Sampler"),
        "patterns": (
            {
                "kick": _step_pattern(0, 6, 8, 10),
                "snare": _step_pattern(4, 12),
                "hat": _step_pattern(0, 2, 4, 6, 8, 10, 12, 14),
                "open_hat": _step_pattern(14),
                "808": _step_pattern(0, 8),
            },
            {
                "kick": _step_pattern(0, 3, 8, 11),
                "snare": _step_pattern(4, 12),
                "hat": _step_pattern(0, 2, 4, 6, 8, 10, 12, 14),
                "open_hat": _step_pattern(7),
                "808": _step_pattern(0, 8),
            },
        ),
    },
    "Lofi Beats": {
        "bpm_range": (70, 92),
        "swing": 0.24,
        "scales": ("Pentatonic", "Minor", "Major", "Dorian"),
        "melody_instruments": ("Electric Piano", "Bell Pluck", "Piano"),
        "patterns": (
            {
                "kick": _step_pattern(0, 7, 8, 14),
                "snare": _step_pattern(4, 12),
                "hat": _step_pattern(0, 3, 6, 8, 11, 14),
                "open_hat": _step_pattern(15),
                "808": _step_pattern(0, 8),
            },
            {
                "kick": _step_pattern(0, 5, 8, 11),
                "snare": _step_pattern(4, 12),
                "hat": _step_pattern(0, 2, 5, 8, 10, 13),
                "open_hat": _step_pattern(7),
                "808": _step_pattern(0, 8),
            },
        ),
    },
    "Modern Pop / R&B": {
        "bpm_range": (96, 132),
        "swing": 0.1,
        "scales": ("Major", "Minor", "Dorian", "Pentatonic"),
        "melody_instruments": ("Lead Synth", "Digital Piano", "Pluck Synth"),
        "patterns": (
            {
                "kick": _step_pattern(0, 4, 8, 10, 12),
                "snare": _step_pattern(4, 12),
                "hat": _step_pattern(0, 2, 4, 6, 8, 10, 12, 14),
                "open_hat": _step_pattern(7, 15),
                "808": _step_pattern(0, 8, 12),
            },
            {
                "kick": _step_pattern(0, 3, 8, 11),
                "snare": _step_pattern(4, 12),
                "hat": _step_pattern(0, 2, 4, 6, 8, 10, 12, 14),
                "open_hat": _step_pattern(3, 11),
                "808": _step_pattern(0, 8),
            },
        ),
    },
}

logo_col, title_col = st.columns([1, 8])
with logo_col:
    st.image("logo.png", width=100)
with title_col:
    st.title("SelfBeats AI - Ultimate Mega Studio")

mode = st.radio("Select Workflow:", ["⚡ 1-Click Auto", "🎼 Pure Instruments Only", "🎛️ Full Hardware Rack (Step-by-Step)"], horizontal=True)
beat_style = st.selectbox("Beat Style:", BEAT_STYLES, index=0)

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


def resonant_lowpass_filter(
    data,
    cutoff=7200.0,
    resonance=0.12,
    fs=STANDARD_SAMPLE_RATE,
):
    """Low-pass an organic voice while retaining a restrained filter resonance."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    if len(data) < 8:
        return data
    cutoff = min(float(cutoff), fs * 0.45)
    lowpass = signal.butter(2, cutoff, btype="lowpass", fs=fs, output="sos")
    resonant_band = signal.butter(
        2,
        [max(80.0, cutoff * 0.72), cutoff * 0.98],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    filtered = signal.sosfilt(lowpass, data)
    resonance_signal = signal.sosfilt(resonant_band, data)
    return np.asarray(
        filtered + resonance_signal * INTERNAL_DTYPE(resonance),
        dtype=INTERNAL_DTYPE,
    )


def band_limit_filter(
    data,
    fs=STANDARD_SAMPLE_RATE,
    low_cutoff=20.0,
    high_cutoff=18000.0,
):
    """Remove subsonic and harsh digital frequencies on every audio path."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    if len(data) < 8:
        return data
    highpass = signal.butter(6, low_cutoff, btype="highpass", fs=fs, output="sos")
    lowpass = signal.butter(6, high_cutoff, btype="lowpass", fs=fs, output="sos")
    filtered = signal.sosfilt(highpass, data)
    filtered = signal.sosfilt(lowpass, filtered)
    return np.asarray(filtered, dtype=INTERNAL_DTYPE)


def presence_boost(
    data,
    fs=STANDARD_SAMPLE_RATE,
    low_cutoff=2500.0,
    high_cutoff=5000.0,
    gain_db=PRESENCE_BOOST_DB,
):
    """Add a restrained presence-band lift for melody and vocal-like detail."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    if len(data) < 8:
        return data
    presence_filter = signal.butter(
        2,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    presence_band = signal.sosfilt(presence_filter, data)
    linear_boost = INTERNAL_DTYPE(10 ** (gain_db / 20.0) - 1.0)
    return np.asarray(data + presence_band * linear_boost, dtype=INTERNAL_DTYPE)


def body_reverb(data, delay_ms=35, decay=0.28, fs=STANDARD_SAMPLE_RATE):
    delay_samples = max(1, int((delay_ms / 1000.0) * fs))
    output = np.array(data, dtype=INTERNAL_DTYPE, copy=True)
    for buffer_start in range(0, len(data), AUDIO_BUFFER_SIZE):
        buffer_end = min(buffer_start + AUDIO_BUFFER_SIZE, len(data))
        first_sample = max(delay_samples, buffer_start)
        for i in range(first_sample, buffer_end):
            output[i] += INTERNAL_DTYPE(output[i - delay_samples] * decay)
    return output


def spatial_reverb_delay(data, fs=STANDARD_SAMPLE_RATE, wet=0.2):
    """Create warm early reflections, pre-delay, and a diffused reverb tail."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    if len(data) < 16:
        return data

    reflections = np.zeros(len(data), dtype=INTERNAL_DTYPE)
    for delay_ms, gain in (
        (17.0, 0.22),
        (29.0, 0.16),
        (43.0, 0.12),
        (67.0, 0.085),
        (97.0, 0.055),
    ):
        delay_samples = max(1, int(delay_ms * fs / 1000.0))
        if delay_samples < len(data):
            reflections[delay_samples:] += data[:-delay_samples] * INTERNAL_DTYPE(gain)

    diffused_tail = body_reverb(
        reflections,
        delay_ms=31.0,
        decay=0.24,
        fs=fs,
    )
    wet_signal = lowpass_filter(
        reflections * INTERNAL_DTYPE(0.65) + diffused_tail * INTERNAL_DTYPE(0.35),
        cutoff=8500.0,
        fs=fs,
    )
    wet = INTERNAL_DTYPE(wet)
    return np.asarray(
        data * (INTERNAL_DTYPE(1.0) - wet * INTERNAL_DTYPE(0.35))
        + wet_signal * wet,
        dtype=INTERNAL_DTYPE,
    )


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


def soft_knee_saturator(
    data,
    threshold=MASTER_LIMITER_THRESHOLD,
    drive=1.15,
):
    """Smoothly round peaks around the -0.3 dBFS saturation threshold."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    threshold = INTERNAL_DTYPE(threshold)
    drive = INTERNAL_DTYPE(drive)
    saturated = threshold * np.tanh((data / threshold) * drive)
    saturated /= np.tanh(drive)
    return np.asarray(saturated, dtype=INTERNAL_DTYPE)


def transparent_peak_limiter(data, ceiling=MASTER_OUTPUT_PEAK):
    """Attenuate overs transparently without adding gain or hard edge clipping."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    ceiling = INTERNAL_DTYPE(ceiling)
    peak = np.max(np.abs(data)) if data.size else INTERNAL_DTYPE(0.0)
    if peak > ceiling:
        data = data * (ceiling / peak)
    return np.clip(data, -ceiling, ceiling).astype(INTERNAL_DTYPE)


def multiband_peak_limiter(
    data,
    fs=STANDARD_SAMPLE_RATE,
    ceiling=MASTER_LIMITER_THRESHOLD,
):
    """Limit low, presence, and high bands separately before the final guard."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    if len(data) < 8:
        return data
    low_band = signal.butter(4, 180.0, btype="lowpass", fs=fs, output="sos")
    presence_band = signal.butter(
        4,
        [180.0, 5000.0],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    high_band = signal.butter(4, 5000.0, btype="highpass", fs=fs, output="sos")
    bands = (
        signal.sosfilt(low_band, data),
        signal.sosfilt(presence_band, data),
        signal.sosfilt(high_band, data),
    )
    limited_bands = [
        transparent_peak_limiter(band, ceiling=ceiling)
        for band in bands
    ]
    summed = np.sum(
        np.asarray(limited_bands, dtype=INTERNAL_DTYPE),
        axis=0,
        dtype=INTERNAL_DTYPE,
    )
    return transparent_peak_limiter(summed, ceiling=ceiling)


def measure_signal(data):
    """Return float64 RMS and peak measurements without changing the signal."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    if data.size == 0:
        return INTERNAL_DTYPE(0.0), INTERNAL_DTYPE(0.0)
    rms = np.sqrt(np.mean(np.square(data), dtype=INTERNAL_DTYPE))
    peak = np.max(np.abs(data))
    return INTERNAL_DTYPE(rms), INTERNAL_DTYPE(peak)


def peak_normalize(data, target_peak=MASTER_OUTPUT_PEAK):
    """Normalize a protected master to the requested final dBFS peak."""
    data = np.asarray(data, dtype=INTERNAL_DTYPE)
    peak = np.max(np.abs(data)) if data.size else INTERNAL_DTYPE(0.0)
    if peak <= 0.0:
        return data
    normalized = data * (INTERNAL_DTYPE(target_peak) / peak)
    return np.clip(normalized, -1.0, 1.0).astype(INTERNAL_DTYPE)


def wav_to_mp3(wav_bytes, bitrate="320k"):
    """Encode WAV bytes to MP3 through an in-memory ffmpeg pipe."""
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [
                ffmpeg_exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-vn",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                bitrate,
                "-ar",
                str(STANDARD_SAMPLE_RATE),
                "-f",
                "mp3",
                "pipe:1",
            ],
            input=wav_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError("MP3 conversion is unavailable in this environment.") from exc

    if result.returncode != 0 or not result.stdout:
        error_message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"MP3 conversion failed: {error_message or 'ffmpeg returned no audio.'}")
    return result.stdout


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
        return resonant_lowpass_filter(
            saw * bow_env,
            cutoff=2900.0,
            resonance=0.08,
            fs=fs,
        )

    elif any(
        voice in inst
        for voice in ["Synth", "Pluck", "Bell", "Sampler", "Vocoder"]
    ):
        vibrato = INTERNAL_DTYPE(1.0) + INTERNAL_DTYPE(0.0025) * np.sin(
            2 * np.pi * 5.0 * t
        )
        phase = np.cumsum(INTERNAL_DTYPE(freq) * vibrato) / INTERNAL_DTYPE(fs)
        phase_cycle = phase % INTERNAL_DTYPE(1.0)
        saw = INTERNAL_DTYPE(2.0) * phase_cycle - INTERNAL_DTYPE(1.0)
        triangle = INTERNAL_DTYPE(2.0) * np.abs(
            INTERNAL_DTYPE(2.0) * phase_cycle - INTERNAL_DTYPE(1.0)
        ) - INTERNAL_DTYPE(1.0)
        shimmer = np.sin(2 * np.pi * phase * 2.0)
        organic_wave = (
            INTERNAL_DTYPE(0.48) * saw
            + INTERNAL_DTYPE(0.38) * triangle
            + INTERNAL_DTYPE(0.14) * shimmer
        )
        decay = INTERNAL_DTYPE(4.2 if "Pluck" in inst or "Bell" in inst else 2.2)
        voice_env = np.exp(-decay * t)
        cutoff = 5200.0 if "Bell" in inst else 6800.0
        return resonant_lowpass_filter(
            organic_wave * voice_env,
            cutoff=cutoff,
            resonance=0.14,
            fs=fs,
        )

    elif any(b in inst for b in ["808", "Sub Bass", "Bass Synth"]):
        start_freq = max(INTERNAL_DTYPE(freq), INTERNAL_DTYPE(32.7))
        pitch = start_freq * np.exp(-INTERNAL_DTYPE(4.5) * t)
        phase = np.cumsum(pitch) / INTERNAL_DTYPE(fs)
        sub = np.sin(2 * np.pi * phase)
        harmonic = INTERNAL_DTYPE(0.16) * np.sin(2 * np.pi * 2 * phase)
        envelope = np.exp(-INTERNAL_DTYPE(2.8) * t)
        return lowpass_filter((sub + harmonic) * envelope, cutoff=180, fs=fs)

    elif any(k in inst for k in ["Piano", "Keyboard"]):
        harmonics = (
            1.0 * np.sin(2 * np.pi * freq * t) * np.exp(-2.8 * t)
            + 0.45 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-4.5 * t)
            + 0.2 * np.sin(2 * np.pi * freq * 3 * t) * np.exp(-7.0 * t)
        )
        return resonant_lowpass_filter(
            harmonics,
            cutoff=6200.0,
            resonance=0.06,
            fs=fs,
        )

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
            "Clap",
            "Hat",
        ]
    ):
        if "Kick" in inst:
            return np.sin(
                2 * np.pi * (140 * np.exp(-38 * t) + 38) * t
            ).astype(INTERNAL_DTYPE) * np.exp(-7 * t)
        else:
            decay = (
                90
                if any(
                    x in inst
                    for x in [
                        "Hi-Hat", "Closed Hi-Hat", "Open Hat", "Hat",
                        "Shaker", "Tambourine", "Triangle", "Maracas",
                    ]
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


def _mutate_steps(pattern, rng, add_probability=0.08, remove_probability=0.03):
    """Make a small, style-safe variation without destroying the downbeat."""
    mutated = np.asarray(pattern, dtype=np.int8).copy()
    for step in range(16):
        if step in (0, 4, 8, 12) and mutated[step]:
            continue
        if mutated[step] and rng.random() < remove_probability:
            mutated[step] = 0
        elif not mutated[step] and rng.random() < add_probability:
            mutated[step] = 1
    return mutated


def _event(
    instrument,
    frequency,
    start_sec,
    length_sec,
    gain,
    rng=None,
    timing_jitter=0.0,
    velocity_jitter=0.0,
):
    if rng is not None:
        start_sec += rng.uniform(-timing_jitter, timing_jitter)
        gain *= rng.uniform(
            max(0.0, 1.0 - velocity_jitter),
            1.0 + velocity_jitter,
        )
    return (
        instrument,
        INTERNAL_DTYPE(frequency),
        INTERNAL_DTYPE(start_sec),
        INTERNAL_DTYPE(length_sec),
        INTERNAL_DTYPE(gain),
    )


def midi_to_frequency(note_number):
    return INTERNAL_DTYPE(440.0 * (2.0 ** ((note_number - 69.0) / 12.0)))


def _scale_pitch(root_midi, intervals, degree, octave_offset=0):
    octave, index = divmod(int(degree), len(intervals))
    return INTERNAL_DTYPE(
        root_midi + intervals[index] + 12 * (octave + octave_offset)
    )


def generate_beat_events(style, bpm, bars, rng, bass_midi_by_bar):
    """Create drum, hi-hat-roll, open-hat, and 808 events for a style."""
    profile = BEAT_STYLE_PROFILES[style]
    selected = rng.choice(profile["patterns"])
    patterns = {
        name: _mutate_steps(
            values,
            rng,
            add_probability=0.12 if name == "hat" else 0.06,
            remove_probability=0.025,
        )
        for name, values in selected.items()
    }
    step_sec = INTERNAL_DTYPE(60.0 / bpm / 4.0)
    swing = INTERNAL_DTYPE(profile["swing"])
    events = []

    for bar in range(bars):
        bar_start = INTERNAL_DTYPE(bar * 16) * step_sec
        bass_frequency = midi_to_frequency(bass_midi_by_bar[bar % len(bass_midi_by_bar)])
        for step in range(16):
            start = bar_start + INTERNAL_DTYPE(step) * step_sec
            if step % 2:
                start += swing * step_sec
            timing_jitter = min(0.008, float(step_sec) * 0.04)
            if patterns["kick"][step]:
                events.append(
                    _event(
                        "Punchy Kick",
                        55.0,
                        start,
                        0.26,
                        0.62,
                        rng=rng,
                        timing_jitter=timing_jitter,
                        velocity_jitter=0.1,
                    )
                )
            if patterns["snare"][step]:
                snare_name = "Crisp Clap" if style == "Modern Pop / R&B" else "Crisp Snare"
                events.append(
                    _event(
                        snare_name,
                        180.0,
                        start,
                        0.22,
                        0.38,
                        rng=rng,
                        timing_jitter=timing_jitter,
                        velocity_jitter=0.12,
                    )
                )
            if patterns["hat"][step]:
                events.append(
                    _event(
                        "Closed Hi-Hat",
                        5000.0,
                        start,
                        0.075,
                        0.2,
                        rng=rng,
                        timing_jitter=timing_jitter,
                        velocity_jitter=0.18,
                    )
                )
            if patterns["open_hat"][step]:
                events.append(
                    _event(
                        "Open Hat",
                        6500.0,
                        start,
                        0.2,
                        0.22,
                        rng=rng,
                        timing_jitter=timing_jitter,
                        velocity_jitter=0.16,
                    )
                )
            if patterns["808"][step]:
                events.append(
                    _event(
                        "808 Bass",
                        bass_frequency,
                        start,
                        0.55,
                        0.48,
                        rng=rng,
                        timing_jitter=timing_jitter * 0.5,
                        velocity_jitter=0.08,
                    )
                )

        if style in ("Hip-Hop / Trap", "Modern Drill") and rng.random() < 0.9:
            roll_start = rng.choice((3, 7, 11, 15))
            for roll_step in range(3):
                start = bar_start + (roll_start + roll_step * 0.5) * step_sec
                events.append(
                    _event(
                        "Closed Hi-Hat",
                        5600.0,
                        start,
                        0.045,
                        0.14,
                        rng=rng,
                        timing_jitter=min(0.003, float(step_sec) * 0.02),
                        velocity_jitter=0.2,
                    )
                )

    return events


def generate_melody_events(
    style,
    bpm,
    bars,
    rng,
    root_midi,
    scale_name,
    progression,
    progression_name="",
):
    """Generate key-aligned harmony plus alternating lead calls and responses."""
    profile = BEAT_STYLE_PROFILES[style]
    intervals = SCALE_INTERVALS[scale_name]
    bar_sec = INTERNAL_DTYPE(60.0 / bpm * 4.0)
    eighth_sec = bar_sec / INTERNAL_DTYPE(8.0)
    lead_instrument = rng.choice(profile["melody_instruments"])
    harmony_instrument = "Piano" if style == "Boom Bap" else "Electric Piano"
    events = []

    for bar in range(bars):
        bar_start = INTERNAL_DTYPE(bar) * bar_sec
        chord_degree = progression[bar % len(progression)]
        chord_degrees = (chord_degree, chord_degree + 2, chord_degree + 4)
        for chord_index, degree in enumerate(chord_degrees):
            chord_pitch = _scale_pitch(root_midi, intervals, degree, octave_offset=1)
            events.append(
                _event(
                    harmony_instrument,
                    midi_to_frequency(chord_pitch),
                    bar_start,
                    bar_sec * INTERNAL_DTYPE(0.9),
                    INTERNAL_DTYPE(0.13 if chord_index == 0 else 0.09),
                    rng=rng,
                    timing_jitter=min(0.008, float(bar_sec) * 0.012),
                    velocity_jitter=0.08,
                )
            )

        is_call = bar % 2 == 0
        motif = rng.choice(MELODY_MOTIFS if is_call else RESPONSE_MOTIFS)
        phrase_instrument = lead_instrument if is_call else harmony_instrument
        phrase_octave = 2 if is_call else 1
        phrase_spacing = eighth_sec if is_call else bar_sec / INTERNAL_DTYPE(4.0)
        for slot, motif_degree in enumerate(motif):
            degree_offset = int(motif_degree)
            pitch = _scale_pitch(
                root_midi,
                intervals,
                chord_degree + degree_offset,
                octave_offset=phrase_octave,
            )
            start = bar_start + INTERNAL_DTYPE(slot) * phrase_spacing
            if slot % 2:
                start += INTERNAL_DTYPE(profile["swing"]) * phrase_spacing
            events.append(
                _event(
                    phrase_instrument,
                    midi_to_frequency(pitch),
                    start,
                    0.24 if is_call else 0.3,
                    INTERNAL_DTYPE(0.18 if is_call else 0.13),
                    rng=rng,
                    timing_jitter=min(
                        0.012,
                        float(phrase_spacing) * 0.035,
                    ),
                    velocity_jitter=0.14,
                )
            )

    return events


def generate_modern_beat_plan(style, duration_sec, rng):
    """Return a key-aware, emotionally coherent beat/melody event plan."""
    profile = BEAT_STYLE_PROFILES[style]
    bpm = rng.randint(*profile["bpm_range"])
    bars = max(1, int(np.ceil(duration_sec / (60.0 / bpm * 4.0))))
    scale_name = rng.choice(profile["scales"])
    intervals = SCALE_INTERVALS[scale_name]
    root_midi = rng.choice((36, 38, 40, 41, 43, 45, 47))
    tonal_mode = (
        "Major"
        if scale_name in ("Major", "Dorian")
        or (scale_name == "Pentatonic" and rng.random() > 0.5)
        else "Minor"
    )
    progression_name, progression = rng.choice(CHORD_PROGRESSIONS[tonal_mode])
    bass_midi_by_bar = [
        _scale_pitch(root_midi, intervals, progression[bar % len(progression)])
        for bar in range(bars)
    ]
    events = generate_beat_events(
        style,
        bpm,
        bars,
        rng,
        bass_midi_by_bar,
    )
    events.extend(
        generate_melody_events(
            style,
            bpm,
            bars,
            rng,
            root_midi,
            scale_name,
            progression,
            progression_name=progression_name,
        )
    )
    return {
        "events": events,
        "bpm": bpm,
        "bars": bars,
        "scale": scale_name,
        "key_mode": tonal_mode,
        "progression_name": progression_name,
        "root_midi": root_midi,
    }


def _add_note_to_layer(layer, instrument, frequency, start_sec, length_sec, gain, fs):
    """Render one event into a float64 layer with safe clipping at the buffer edge."""
    start_index = max(0, int(round(float(start_sec) * fs)))
    if start_index >= len(layer):
        return
    sound = synthesize_hifi_sound(
        instrument,
        frequency,
        max(0.03, float(length_sec)),
        fs,
    )
    available = min(len(sound), len(layer) - start_index)
    if available:
        layer[start_index : start_index + available] += (
            sound[:available] * INTERNAL_DTYPE(gain)
        )


def _needs_spatial_depth(instrument):
    return not any(
        marker in instrument
        for marker in [
            "Kick",
            "Snare",
            "Clap",
            "Hat",
            "Drum",
            "808",
            "Bass",
        ]
    )


def generate_track(
    inst_list,
    fx_list,
    is_auto=False,
    duration=15,
    beat_style="Hip-Hop / Trap",
    use_beat_engine=True,
):
    fs = STANDARD_SAMPLE_RATE
    seed = int(time.time() * 1000) ^ random.randint(1000, 999999)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    rng = random.Random(seed)

    total_samples = int(duration * fs)
    total_signal = np.zeros(total_samples, dtype=INTERNAL_DTYPE)
    number_of_active_instruments = 0
    layer_rms_values = []
    layer_peak_values = []
    rendered_layers = {}

    if is_auto:
        inst_list = rng.sample(pure_instruments, k=rng.randint(3, 6))

    for inst in inst_list:
        layer = np.zeros(total_samples, dtype=INTERNAL_DTYPE)
        bpm = rng.randint(88, 128)
        beat_sec = 60.0 / bpm
        step = int(
            (
                beat_sec
                / (4 if any(p in inst for p in ["Hi-Hat", "Tabla", "Shaker", "Drum"]) else 2)
            )
            * fs
        )
        step = max(1, step)

        for i in range(0, total_samples, step):
            if rng.random() > 0.2:
                freq = rng.choice([130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94, 261.63])
                n_len = rng.choice([0.4, 0.8, 1.2])
                sound = synthesize_hifi_sound(inst, freq, n_len, fs)

                avail = min(len(sound), total_samples - i)
                layer[i : i + avail] += sound[:avail] * INTERNAL_DTYPE(0.35)

        rendered_layers[inst] = layer

    if use_beat_engine:
        plan = generate_modern_beat_plan(beat_style, duration, rng)
        for instrument, frequency, start_sec, length_sec, gain in plan["events"]:
            if instrument not in rendered_layers:
                rendered_layers[instrument] = np.zeros(
                    total_samples,
                    dtype=INTERNAL_DTYPE,
                )
            _add_note_to_layer(
                rendered_layers[instrument],
                instrument,
                frequency,
                start_sec,
                length_sec,
                gain,
                fs,
            )

    for instrument, layer in list(rendered_layers.items()):
        if _needs_spatial_depth(instrument):
            rendered_layers[instrument] = spatial_reverb_delay(
                layer,
                fs=fs,
                wet=0.2,
            )

    for layer in rendered_layers.values():
        layer_rms, layer_peak = measure_signal(layer)
        if layer_peak > 0.0:
            total_signal += layer
            number_of_active_instruments += 1
            layer_rms_values.append(layer_rms)
            layer_peak_values.append(layer_peak)

    # Estimate mix energy from each layer before the layers are summed.
    if layer_rms_values:
        pre_mix_rms = np.sqrt(
            np.sum(
                np.square(np.asarray(layer_rms_values, dtype=INTERNAL_DTYPE)),
                dtype=INTERNAL_DTYPE,
            )
        )
        pre_mix_peak = np.max(
            np.asarray(layer_peak_values, dtype=INTERNAL_DTYPE)
        )
    else:
        pre_mix_rms = INTERNAL_DTYPE(0.0)
        pre_mix_peak = INTERNAL_DTYPE(0.0)
    active_track_divisor = max(
        INTERNAL_DTYPE(1.0),
        INTERNAL_DTYPE(number_of_active_instruments) * INTERNAL_DTYPE(0.7),
    )
    rms_guard = max(INTERNAL_DTYPE(1.0), pre_mix_rms / INTERNAL_DTYPE(0.5))
    peak_guard_scale = max(INTERNAL_DTYPE(1.0), pre_mix_peak / MASTER_LIMITER_THRESHOLD)

    # Reduce the mix by 0.5 and by active-track count before saturation.
    master_signal = (
        total_signal
        * INTERNAL_DTYPE(0.5)
        / active_track_divisor
        / max(rms_guard, peak_guard_scale)
    )
    master_signal *= MASTER_HEADROOM_GAIN * MASTER_GAIN

    if any(
        f in fx_list
        for f in ["Distortion Pedal", "Overdrive Pedal", "Fuzz Pedal", "Saturation Unit"]
    ):
        master_signal = soft_limiter(
            master_signal * INTERNAL_DTYPE(1.7),
            threshold=0.75,
            drive=1.1,
        )

    master_signal = body_reverb(master_signal, fs=fs)
    master_signal = band_limit_filter(
        master_signal,
        fs=fs,
        low_cutoff=30.0,
        high_cutoff=16000.0,
    )
    master_signal = presence_boost(
        master_signal,
        fs=fs,
        low_cutoff=2500.0,
        high_cutoff=5000.0,
        gain_db=PRESENCE_BOOST_DB,
    )
    master_signal = multiband_peak_limiter(
        master_signal,
        fs=fs,
        ceiling=MASTER_LIMITER_THRESHOLD,
    )
    master_signal = soft_knee_saturator(
        master_signal,
        threshold=MASTER_LIMITER_THRESHOLD,
        drive=1.15,
    )
    master_signal = band_limit_filter(
        master_signal,
        fs=fs,
        low_cutoff=30.0,
        high_cutoff=16000.0,
    )
    master = transparent_peak_limiter(
        master_signal,
        ceiling=MASTER_OUTPUT_PEAK,
    )
    master = peak_normalize(master, target_peak=MASTER_OUTPUT_PEAK)
    master = transparent_peak_limiter(master, ceiling=MASTER_OUTPUT_PEAK)
    byte_io = io.BytesIO()
    sf.write(byte_io, master, fs, format="WAV", subtype="PCM_16")
    return byte_io.getvalue(), list(rendered_layers)


render_requested = st.button("🚀 Render Music Track", use_container_width=True)
random_beat_requested = st.button(
    "🎲 Generate Random Melody & Beat",
    use_container_width=True,
)

if render_requested or random_beat_requested:
    is_auto = mode == "⚡ 1-Click Auto" or random_beat_requested
    if not is_auto and not selected_instruments:
        st.warning("⚠️ Please select at least one instrument.")
    else:
        with st.spinner("🎧 Synthesizing DSP Audio..."):
            audio_wav, used = generate_track(
                selected_instruments,
                selected_fx,
                is_auto=is_auto,
                beat_style=beat_style,
                use_beat_engine=True,
            )
            st.success("🎉 Composition Generated Successfully!")
            st.caption(f"Generated with the {beat_style} beat and melody engine.")
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