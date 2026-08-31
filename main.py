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