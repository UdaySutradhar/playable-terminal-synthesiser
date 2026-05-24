# 🎛️ Bhairav Synth OS

A real-time, polyphonic terminal synthesizer and vocal pitch tracker built entirely in Rust. 

This project explores the intersection of low-level systems engineering, real-time digital signal processing (DSP), and Indian classical music. It features a custom lock-free audio engine tuned specifically to the intervals of **Raag Bhairav**, complete with a generative Tanpura drone, dynamic effects graph, and a hardware-accelerated vocal pitch tracker.

## 🚀 Architectural Highlights

This project was built with strict adherence to real-time audio constraints, prioritizing deterministic execution and memory safety.

* **Lock-Free Concurrency (The Try-Lock Pattern):** Bridging a 60Hz UI rendering thread (Crossterm/Ratatui) with a 44,100Hz OS audio thread without causing buffer underruns. Uses `crossbeam` channels for atomic message passing and `try_lock()` to ensure the audio thread never blocks waiting for the UI.
* **Zero Heap Allocations on the Audio Thread:** Features like the Delay/Echo effect are implemented using pre-allocated circular buffers (`Vec` with wrapping indices). Disk I/O (WAV recording) is decoupled via a bounded channel passing fixed-size `[f32; 1024]` arrays to prevent the OS memory allocator from stalling the DSP loop.
* **High-Resolution FFT Pitch Tracking:** Utilizes a 4096-point Fast Fourier Transform to analyze microphone input in real-time. Implements DSP **Hysteresis** (hold timers) and an **Exponential Moving Average (EMA)** to filter out mathematical jitter and provide buttery-smooth vocal pitch tracking.
* **Hardware MIDI Integration:** Wakes up a dedicated background thread to read raw USB MIDI hex bytes, dynamically allocating physical hardware inputs to the software polyphony pool.

## 🧰 Core Features

* **Polyphonic Resource Pool:** Dynamically allocates voices using a custom Attack/Decay "Pluck" envelope to prevent audio popping.
* **Modular Effects Graph:** Includes a mathematically modeled LFO-modulated Low-Pass filter (Wah-Wah) and a feedback Ring-Buffer Delay.
* **Generative & Rhythmic Threads:** Features an algorithmic Tanpura background drone and a mathematically precise BPM Metronome thread.
* **Asynchronous Disk I/O:** Streams live 32-bit float audio directly to a `.wav` file on disk without interrupting the audio thread.
* **TUI Oscilloscope & Spectrum Analyzer:** Real-time visual feedback of time-domain (Waveform) and frequency-domain (EQ) data directly in the terminal.

## 🎮 Controls (Home Row Mapping)
Make sure your terminal is focused. Notes are mapped to the Shuddha and Komal intervals of Raag Bhairav.
Key | Action / Note | Frequency(Hz)<br>
A       Sa (C4)           261.63<br>
S       Komal Re (Db4)    277.18<br>
D       Shuddha Ga (E4)   329.63<br>
F       Shuddha Ma (F4)   349.23<br>
G       Pa (G4)           392.00<br>
H       Komal Dha (Ab4)   415.30<br>
J       Shuddha Ni (B4)   493.88<br>
K       Sa' (C5)          523.25

## System Toggles
>1, 2, 3 - Switch Timbre (Sine, Square, Sawtooth)
>T - Toggle Generative Tanpura Drone
>V - Toggle LFO Low-Pass Filter
>B - Toggle BPM Metronome Clock
>[ / ] - Decrease / Increase Tempo (BPM)
>R - Start/Stop WAV Recording (Saves as synth_recording.wav)
>ESC - Gracefully exit and release audio streams.

## 🎤 Vocal Pitch Tracker
Upon launching, the engine will request microphone permissions. Sing into your microphone, and the engine will run a high-resolution FFT over your vocal input, tracking your fundamental pitch and snapping it to the closest interval in the Bhairav scale.


## 🛠️ Installation & Usage

**Prerequisites:** Ensure you have [Rust and Cargo](https://rustup.rs/) installed. *(Linux users may need `libasound2-dev` for ALSA MIDI support).*

```bash
git clone [https://github.com/yourusername/bhairav-synth-os.git](https://github.com/yourusername/bhairav-synth-os.git)
cd bhairav-synth-os
cargo run --release
