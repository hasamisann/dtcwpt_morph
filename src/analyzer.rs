use rustfft::{FftPlanner, num_complex::Complex, num_traits::Zero};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU32, Ordering};

/// FFT size for spectrum analysis (1024 samples ≈ 23ms at 44.1kHz)
pub const ANALYZER_FFT_SIZE: usize = 2048;

/// Number of frequency bins (FFT_SIZE / 2 + 1)
pub const ANALYZER_NUM_BINS: usize = ANALYZER_FFT_SIZE / 2 + 1;

/// Shared data between audio and GUI threads.
pub struct SharedAnalyzerData {
    /// Ring buffer for input samples (L+R averaged to mono)
    pub input_buffer: Mutex<Vec<f32>>,
    /// Ring buffer for output samples (L+R averaged to mono)
    pub output_buffer: Mutex<Vec<f32>>,
    /// Write cursor for input buffer
    pub input_cursor: AtomicU32,
    /// Write cursor for output buffer
    pub output_cursor: AtomicU32,
    /// Sample rate (for GUI reference)
    pub sample_rate: AtomicU32,
}

impl SharedAnalyzerData {
    /// Create new shared analyzer data.
    pub fn new() -> Self {
        Self {
            input_buffer: Mutex::new(vec![0.0; ANALYZER_FFT_SIZE]),
            output_buffer: Mutex::new(vec![0.0; ANALYZER_FFT_SIZE]),
            input_cursor: AtomicU32::new(0),
            output_cursor: AtomicU32::new(0),
            sample_rate: AtomicU32::new(44100),
        }
    }

    /// Push input and output samples from audio thread.
    /// Samples are averaged to mono and written to ring buffers.
    /// 
    /// # Arguments
    /// * `input_l`, `input_r` - Input stereo samples (f64 from DSP)
    /// * `output_l`, `output_r` - Output stereo samples (f64 from DSP)
    pub fn push_samples(
        &self,
        input_l: &[f64],
        input_r: &[f64],
        output_l: &[f64],
        output_r: &[f64],
    ) {
        // --- Input Buffer ---
        if let Ok(mut buf) = self.input_buffer.lock() {
            let mut cursor = self.input_cursor.load(Ordering::Acquire) as usize;
            let len = buf.len();
            
            for i in 0..input_l.len() {
                // Average to mono and convert to f32
                let sample = ((input_l[i] + input_r[i]) * 0.5) as f32;
                buf[cursor] = sample;
                
                cursor += 1;
                if cursor >= len {
                    cursor = 0;
                }
            }
            self.input_cursor.store(cursor as u32, Ordering::Release);
        }

        // --- Output Buffer ---
        if let Ok(mut buf) = self.output_buffer.lock() {
            let mut cursor = self.output_cursor.load(Ordering::Acquire) as usize;
            let len = buf.len();

            for i in 0..output_l.len() {
                let sample = ((output_l[i] + output_r[i]) * 0.5) as f32;
                buf[cursor] = sample;
                
                cursor += 1;
                if cursor >= len {
                    cursor = 0;
                }
            }
            self.output_cursor.store(cursor as u32, Ordering::Release);
        }
    }

    /// Set the sample rate (called during initialize).
    pub fn set_sample_rate(&self, sample_rate: f32) {
        self.sample_rate.store(sample_rate as u32, Ordering::Relaxed);
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate.load(Ordering::Relaxed) as f32
    }
}

/// Spectrum analyzer for GUI thread.
/// Performs FFT and smoothing on data from SharedAnalyzerData.
pub struct SpectrumAnalyzer {
    /// Reference to shared data
    shared: Arc<SharedAnalyzerData>,
    /// FFT planner (reusable)
    fft: Arc<dyn rustfft::Fft<f32>>,
    /// Hann window coefficients
    window: Vec<f32>,
    /// Scratch buffer for FFT input
    fft_input: Vec<Complex<f32>>,
    /// Scratch buffer for FFT output
    fft_output: Vec<Complex<f32>>,
    /// Smoothed input magnitudes (dB)
    pub smoothed_input: Vec<f32>,
    /// Smoothed output magnitudes (dB)
    pub smoothed_output: Vec<f32>,
    /// Smoothing factor (0.0 = no smoothing, 1.0 = freeze)
    pub smoothing: f32,
}

impl SpectrumAnalyzer {
    /// Create a new spectrum analyzer.
    /// 
    /// # Arguments
    /// * `shared` - Arc to shared analyzer data
    /// * `smoothing` - Smoothing factor (recommend 0.8-0.9)
    pub fn new(shared: Arc<SharedAnalyzerData>, smoothing: f32) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(ANALYZER_FFT_SIZE);

        // Precompute Hann window
        let mut window = vec![0.0; ANALYZER_FFT_SIZE];
        for i in 0..ANALYZER_FFT_SIZE {
            let t = 2.0 * std::f32::consts::PI * i as f32 / ((ANALYZER_FFT_SIZE - 1) as f32);
            window[i] = 0.5 * (1.0 - t.cos());
        }

        Self {
            shared,
            fft,
            window,
            fft_input: vec![Complex::zero(); ANALYZER_FFT_SIZE],
            fft_output: vec![Complex::zero(); ANALYZER_FFT_SIZE],
            smoothed_input: vec![-80.0; ANALYZER_NUM_BINS],
            smoothed_output: vec![-80.0; ANALYZER_NUM_BINS],
            smoothing,
        }
    }

    /// Compute spectrum and return (input_mags_db, output_mags_db).
    /// Each vector has ANALYZER_NUM_BINS elements.
    /// Values are in dB, clamped to [-80, 0].
    pub fn compute(&mut self) -> (&[f32], &[f32]) {
        self.process_buffer(true);  // Input
        self.process_buffer(false); // Output
        (&self.smoothed_input, &self.smoothed_output)
    }

    fn process_buffer(&mut self, is_input: bool) {
        // 1. Copy data from shared buffer
        {
            let buffer_lock = if is_input {
                self.shared.input_buffer.lock()
            } else {
                self.shared.output_buffer.lock()
            };

            if let Ok(buf) = buffer_lock {
                // Determine start index based on cursor to read linear block?
                // Actually FFT doesn't care about phase shift for magnitude, 
                // but windowing expects a continuous block.
                // Best to read from cursor-N to cursor.
                let cursor = if is_input {
                    self.shared.input_cursor.load(Ordering::Acquire) as usize
                } else {
                    self.shared.output_cursor.load(Ordering::Acquire) as usize
                };
                
                // Read continuous block ending at cursor
                // Since it's a ring buffer, we might need two copies
                for i in 0..ANALYZER_FFT_SIZE {
                     // index = (cursor - N + i) wrapped
                     // To avoid negative modulo: (cursor + len - N + i) % len
                     let idx = (cursor + ANALYZER_FFT_SIZE - ANALYZER_FFT_SIZE + i) % ANALYZER_FFT_SIZE;
                     
                     let sample = buf[idx];
                     let windowed = sample * self.window[i];
                     self.fft_input[i] = Complex::new(windowed, 0.0);
                }
            }
        }

        // 2. FFT
        // RustFFT's process is in-place. Copy input to output, then process output.
        self.fft_output.copy_from_slice(&self.fft_input);
        self.fft.process(&mut self.fft_output);

        // 3. Magnitude & Smoothing
        let target_smooth = if is_input { &mut self.smoothed_input } else { &mut self.smoothed_output };
        
        let norm_factor = 1.0 / (ANALYZER_FFT_SIZE as f32);
        
        for i in 0..ANALYZER_NUM_BINS {
            let re = self.fft_output[i].re;
            let im = self.fft_output[i].im;
            let mag = (re * re + im * im).sqrt() * norm_factor;
            
            // Convert to dB
            let db = 20.0 * (mag + 1e-10).log10();
            let clamped_db = db.clamp(-80.0, 0.0);
            
            // Exponential smoothing
            target_smooth[i] = self.smoothing * target_smooth[i] + (1.0 - self.smoothing) * clamped_db;
        }
    }
}
