use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use crossterm::{
    event::{poll, read, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    prelude::*,
    widgets::{Axis, BarChart, Block, Borders, Chart, Dataset, GraphType, Paragraph},
};
use rustfft::{FftPlanner, num_complex::Complex};
use std::{
    f32::consts::PI,
    io::stdout,
    sync::{atomic::{AtomicBool, AtomicU32, Ordering}, Arc, Mutex},
    thread,
    time::Duration,
};

const MAX_VOICES: usize = 12;
const WAVEFORM_SIZE: usize = 256; 
const CHUNK_SIZE: usize = 1024;

// --- 1. System Enums & State ---

#[derive(Clone, Copy, PartialEq, Eq)]
enum VoiceId {
    Key(char),
    Midi(u8),
    Drone(char),
    Metronome, // NEW: Dedicated ID for our BPM click
}

#[derive(Clone)]
struct UiState {
    voice_levels: [f32; MAX_VOICES],
    waveform: [f32; WAVEFORM_SIZE],
    wave_idx: usize,
    is_recording: bool,
    is_filter_on: bool,
    is_tanpura_on: bool,
    is_bpm_on: bool,
    bpm: u32,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            voice_levels: [0.0; MAX_VOICES],
            waveform: [0.0; WAVEFORM_SIZE],
            wave_idx: 0,
            is_recording: false,
            is_filter_on: false,
            is_tanpura_on: false,
            is_bpm_on: false,
            bpm: 120,
        }
    }
}

enum AudioCommand {
    NoteOn(VoiceId, f32),
    NoteOff(VoiceId),
    SetWaveform(Waveform),
    ToggleFilter,
}

enum RecordMsg { Start(u32), Audio([f32; CHUNK_SIZE]), Stop }

// --- 2. Modular Architecture ---

trait AudioNode: Send { fn process(&mut self, input: f32, sample_rate: f32) -> f32; }

struct LfoFilter { lfo_phase: f32, last_out: f32 }
impl AudioNode for LfoFilter {
    fn process(&mut self, input: f32, sample_rate: f32) -> f32 {
        let lfo_val = (self.lfo_phase * 2.0 * PI).sin() * 0.5 + 0.5;
        let cutoff = 200.0 + (1600.0 * lfo_val);
        self.lfo_phase = (self.lfo_phase + (0.5 / sample_rate)) % 1.0;
        let alpha = (2.0 * PI * cutoff / sample_rate).min(1.0);
        self.last_out += alpha * (input - self.last_out);
        self.last_out
    }
}

struct RingBufferDelay { buffer: Vec<f32>, write_idx: usize, feedback: f32 }
impl RingBufferDelay {
    fn new(sr: f32, ms: f32, fb: f32) -> Self { 
        Self { buffer: vec![0.0; ((sr * ms) / 1000.0).max(1.0) as usize], write_idx: 0, feedback: fb } 
    }
}
impl AudioNode for RingBufferDelay {
    fn process(&mut self, input: f32, _sr: f32) -> f32 {
        let delayed = self.buffer[self.write_idx];
        self.buffer[self.write_idx] = input + (delayed * self.feedback);
        self.write_idx = (self.write_idx + 1) % self.buffer.len();
        input + (delayed * 0.5)
    }
}

// --- 3. DSP Core ---

#[derive(Clone, Copy, PartialEq)]
enum Waveform { Sine, Square, Sawtooth }

#[derive(Copy, Clone, PartialEq)]
enum EnvState { Idle, Attack, Decay }

#[derive(Clone)]
struct Voice { id: VoiceId, frequency: f32, phase: f32, env_level: f32, state: EnvState }

impl Voice {
    fn new() -> Self { Self { id: VoiceId::Key(' '), frequency: 0.0, phase: 0.0, env_level: 0.0, state: EnvState::Idle } }
    fn next_sample(&mut self, sample_rate: f32, waveform: Waveform) -> f32 {
        if self.state == EnvState::Idle { return 0.0; }
        match self.state {
            EnvState::Attack => {
                // Make the metronome attack incredibly fast so it clicks
                let attack_time = if self.id == VoiceId::Metronome { 0.002 } else { 0.01 };
                self.env_level += 1.0 / (attack_time * sample_rate);
                if self.env_level >= 1.0 { self.env_level = 1.0; self.state = EnvState::Decay; }
            }
            EnvState::Decay => {
                // Make the metronome decay incredibly fast (staccato)
                let decay_time = if self.id == VoiceId::Metronome { 0.05 } else { 1.5 };
                self.env_level -= 1.0 / (decay_time * sample_rate);
                if self.env_level <= 0.0 { self.env_level = 0.0; self.state = EnvState::Idle; }
            }
            _ => {}
        }
        let raw = match waveform {
            Waveform::Sine => (self.phase * 2.0 * PI).sin(),
            Waveform::Square => if self.phase < 0.5 { 0.3 } else { -0.3 },
            Waveform::Sawtooth => (self.phase * 2.0) - 1.0,
        };
        self.phase = (self.phase + (self.frequency / sample_rate)) % 1.0;
        raw * self.env_level
    }
}

// --- 4. Main System Run ---

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host.default_output_device().expect("No output device");
    let config = device.default_output_config()?;
    let sample_rate = config.sample_rate().0 as f32;

    let ui_state = Arc::new(Mutex::new(UiState::default()));
    let (tx, rx) = unbounded();
    let (rec_tx, rec_rx) = bounded(50);

    // THREAD: Disk I/O WAV Writer
    thread::spawn(move || {
        let mut writer: Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>> = None;
        while let Ok(msg) = rec_rx.recv() {
            match msg {
                RecordMsg::Start(sr) => {
                    let spec = hound::WavSpec { channels: 1, sample_rate: sr, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
                    writer = Some(hound::WavWriter::create("synth_recording.wav", spec).unwrap());
                }
                RecordMsg::Audio(samples) => { if let Some(w) = &mut writer { for &s in &samples { w.write_sample(s).unwrap_or(()); } } }
                RecordMsg::Stop => { if let Some(w) = writer.take() { w.finalize().unwrap_or(()); } }
            }
        }
    });

    // THREAD: Generative Tanpura
    let tanpura_active = Arc::new(AtomicBool::new(false));
    let t_active_clone = tanpura_active.clone();
    let tx_tanpura = tx.clone();
    thread::spawn(move || {
        loop {
            if t_active_clone.load(Ordering::Relaxed) {
                let _ = tx_tanpura.send(AudioCommand::NoteOn(VoiceId::Drone('S'), 130.81));
                thread::sleep(Duration::from_millis(1500));
                if !t_active_clone.load(Ordering::Relaxed) { continue; }
                let _ = tx_tanpura.send(AudioCommand::NoteOn(VoiceId::Drone('P'), 196.00));
                thread::sleep(Duration::from_millis(1500));
            } else { thread::sleep(Duration::from_millis(100)); }
        }
    });

    // --- NEW THREAD: BPM Clock / Metronome ---
    let bpm_active = Arc::new(AtomicBool::new(false));
    let current_bpm = Arc::new(AtomicU32::new(120));
    let b_active_clone = bpm_active.clone();
    let b_val_clone = current_bpm.clone();
    let tx_bpm = tx.clone();
    
    thread::spawn(move || {
        loop {
            if b_active_clone.load(Ordering::Relaxed) {
                let bpm = b_val_clone.load(Ordering::Relaxed).max(30); // Prevent divide by zero
                let interval_ms = 60_000 / bpm;

                // Send a high-pitched click (A5 = 880Hz)
                let _ = tx_bpm.send(AudioCommand::NoteOn(VoiceId::Metronome, 880.0));
                
                thread::sleep(Duration::from_millis(interval_ms as u64));
            } else {
                thread::sleep(Duration::from_millis(50));
            }
        }
    });

    // THREAD: USB MIDI Hardware Listener
    let tx_midi = tx.clone();
    thread::spawn(move || {
        if let Ok(midi_in) = midir::MidiInput::new("Rust Synth") {
            if let Some(port) = midi_in.ports().first() {
                let _conn_in = midi_in.connect(port, "synth-in", move |_stamp, message, _| {
                    if message.len() >= 3 {
                        let status = message[0] & 0xF0;
                        let note = message[1];
                        let vel = message[2];
                        let freq = 440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0);
                        if status == 0x90 && vel > 0 {
                            let _ = tx_midi.send(AudioCommand::NoteOn(VoiceId::Midi(note), freq));
                        } else if status == 0x80 || (status == 0x90 && vel == 0) {
                            let _ = tx_midi.send(AudioCommand::NoteOff(VoiceId::Midi(note)));
                        }
                    }
                }, ()).unwrap();
                loop { thread::sleep(Duration::from_secs(1)); } 
            }
        }
    });

    // AUDIO THREAD
    let ui_clone = Arc::clone(&ui_state);
    let mut voices = vec![Voice::new(); MAX_VOICES];
    let mut active_waveform = Waveform::Sine;
    
    let delay_node = Box::new(RingBufferDelay::new(sample_rate, 350.0, 0.6));
    let mut filter_node = LfoFilter { lfo_phase: 0.0, last_out: 0.0 };
    let mut fx_chain: Vec<Box<dyn AudioNode>> = vec![delay_node];
    
    let mut filter_on = false;
    let mut recording_on = false;
    let mut rec_buffer = [0.0; CHUNK_SIZE];
    let mut rec_idx = 0;

    let stream = device.build_output_stream(
        &config.into(),
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            while let Ok(cmd) = rx.try_recv() {
                match cmd {
                    AudioCommand::NoteOn(id, freq) => {
                        let target = voices.iter().position(|v| v.state == EnvState::Idle)
                            .unwrap_or_else(|| voices.iter().enumerate().min_by(|(_, a), (_, b)| a.env_level.partial_cmp(&b.env_level).unwrap()).map(|(i, _)| i).unwrap_or(0));
                        voices[target].id = id; voices[target].frequency = freq; voices[target].state = EnvState::Attack;
                    }
                    AudioCommand::NoteOff(id) => {
                        if let Some(v) = voices.iter_mut().find(|v| v.id == id && v.state == EnvState::Attack) { v.state = EnvState::Decay; }
                    }
                    AudioCommand::SetWaveform(w) => active_waveform = w,
                    AudioCommand::ToggleFilter => filter_on = !filter_on,
                }
            }

            let mut state_lock = ui_clone.try_lock();

            for sample in data.iter_mut() {
                let mut mixed = 0.0;
                for v in voices.iter_mut() { mixed += v.next_sample(sample_rate, active_waveform); }
                
                if filter_on { mixed = filter_node.process(mixed, sample_rate); }
                for fx in fx_chain.iter_mut() { mixed = fx.process(mixed, sample_rate); }
                
                let final_out = mixed * 0.15;
                *sample = final_out;

                if recording_on {
                    rec_buffer[rec_idx] = final_out; rec_idx += 1;
                    if rec_idx == CHUNK_SIZE { let _ = rec_tx.try_send(RecordMsg::Audio(rec_buffer)); rec_idx = 0; }
                }

                if let Ok(ref mut state) = state_lock {
                    let idx = state.wave_idx;
                    state.waveform[idx] = final_out;
                    state.wave_idx = (idx + 1) % WAVEFORM_SIZE;
                }
            }

            if let Ok(ref mut state) = state_lock {
                for (i, v) in voices.iter().enumerate() { state.voice_levels[i] = v.env_level; }
                state.is_filter_on = filter_on;
                state.is_recording = recording_on;
            }
        },
        |err| eprintln!("Stream error: {}", err),
        None,
    )?;

    stream.play()?;

    // --- TERMINAL UI LOOP ---
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;
    let mut fft_planner = FftPlanner::new();

    loop {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(4), Constraint::Min(8), Constraint::Length(8), Constraint::Length(8)])
                .split(f.size());

            let state = ui_state.lock().unwrap().clone();

            let header = Paragraph::new(format!(
                "🎛️  Bhairav Synth OS | T: Tanpura ({}) | V: Filter ({}) | R: Record ({}) | ESC to Quit\n\
                 B: Metronome ({}) | Tempo: [ decrease, ] increase ({} BPM)",
                 if state.is_tanpura_on { "ON" } else { "OFF" },
                 if state.is_filter_on { "ON" } else { "OFF" },
                 if state.is_recording { "REC" } else { "OFF" },
                 if state.is_bpm_on { "ON" } else { "OFF" },
                 state.bpm
            )).style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
             .block(Block::default().borders(Borders::ALL).title("Core System Status"));
            f.render_widget(header, chunks[0]);

            let wave_data: Vec<(f64, f64)> = state.waveform.iter().enumerate().map(|(i, &v)| (i as f64, v as f64 * 3.0)).collect();
            let chart = Chart::new(vec![Dataset::default().marker(symbols::Marker::Braille).graph_type(GraphType::Line).style(Style::default().fg(Color::Cyan)).data(&wave_data)])
                .block(Block::default().title("Time Domain: Master Out").borders(Borders::ALL))
                .x_axis(Axis::default().bounds([0.0, WAVEFORM_SIZE as f64]))
                .y_axis(Axis::default().bounds([-1.0, 1.0]).labels(vec!["-1".into(), "0".into(), "1".into()]));
            f.render_widget(chart, chunks[1]);

            let fft = fft_planner.plan_fft_forward(WAVEFORM_SIZE);
            let mut fft_buffer: Vec<Complex<f32>> = state.waveform.iter().map(|&v| Complex { re: v, im: 0.0 }).collect();
            fft.process(&mut fft_buffer);
            
            let mut bands = [0.0; 8];
            for i in 1..128 { 
                let mag = (fft_buffer[i].re.powi(2) + fft_buffer[i].im.powi(2)).sqrt();
                let bin = (i / 16).min(7);
                bands[bin] += mag;
            }
            
            let fft_labels = ["Sub", "Bass", "LoMid", "Mid", "HiMid", "Pres", "Treb", "Air"];
            let fft_bar_data: Vec<(&str, u64)> = bands.iter().enumerate()
                .map(|(i, &mag)| (fft_labels[i], (mag * 2.0).min(100.0) as u64)).collect();
            let fft_chart = BarChart::default().block(Block::default().title("Frequency Domain: FFT Spectrum Analyzer").borders(Borders::ALL))
                .data(&fft_bar_data).bar_width(6).bar_style(Style::default().fg(Color::Magenta)).value_style(Style::default().bg(Color::Magenta));
            f.render_widget(fft_chart, chunks[2]);

            let v_labels = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "T1", "BPM"];
            let bar_data: Vec<(&str, u64)> = state.voice_levels.iter().enumerate().map(|(i, &lvl)| (v_labels[i], (lvl * 100.0) as u64)).collect();
            let barchart = BarChart::default().block(Block::default().title("Polyphony Thread Pool").borders(Borders::ALL))
                .data(&bar_data).bar_width(5).bar_style(Style::default().fg(Color::Green)).value_style(Style::default().bg(Color::Green));
            f.render_widget(barchart, chunks[3]);
        })?;

        if poll(Duration::from_millis(16))? {
            if let Event::Key(key_event) = read()? {
                if key_event.kind == KeyEventKind::Press {
                    match key_event.code {
                        KeyCode::Esc => break,
                        KeyCode::Char('1') => { let _ = tx.send(AudioCommand::SetWaveform(Waveform::Sine)); }
                        KeyCode::Char('2') => { let _ = tx.send(AudioCommand::SetWaveform(Waveform::Square)); }
                        KeyCode::Char('3') => { let _ = tx.send(AudioCommand::SetWaveform(Waveform::Sawtooth)); }
                        
                        KeyCode::Char('t') => { 
                            let current = tanpura_active.load(Ordering::Relaxed);
                            tanpura_active.store(!current, Ordering::Relaxed);
                            ui_state.lock().unwrap().is_tanpura_on = !current;
                        }
                        
                        // NEW BPM CONTROLS
                        KeyCode::Char('b') => { 
                            let current = bpm_active.load(Ordering::Relaxed);
                            bpm_active.store(!current, Ordering::Relaxed);
                            ui_state.lock().unwrap().is_bpm_on = !current;
                        }
                        KeyCode::Char('[') => {
                            let new_bpm = current_bpm.load(Ordering::Relaxed).saturating_sub(5).max(40);
                            current_bpm.store(new_bpm, Ordering::Relaxed);
                            ui_state.lock().unwrap().bpm = new_bpm;
                        }
                        KeyCode::Char(']') => {
                            let new_bpm = current_bpm.load(Ordering::Relaxed).saturating_add(5).min(300);
                            current_bpm.store(new_bpm, Ordering::Relaxed);
                            ui_state.lock().unwrap().bpm = new_bpm;
                        }
                        
                        KeyCode::Char('v') => { let _ = tx.send(AudioCommand::ToggleFilter); }
                        KeyCode::Char(c @ ('a'|'s'|'d'|'f'|'g'|'h'|'j'|'k')) => {
                            let freq = match c { 'a'=>261.63, 's'=>277.18, 'd'=>329.63, 'f'=>349.23, 'g'=>392.00, 'h'=>415.30, 'j'=>493.88, 'k'=>523.25, _=>0.0 };
                            let _ = tx.send(AudioCommand::NoteOn(VoiceId::Key(c), freq));
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    stdout().execute(LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}