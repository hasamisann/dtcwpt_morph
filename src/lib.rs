use nih_plug::prelude::*;
use nih_plug_egui::EguiState;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

pub mod knob;
mod constants;
mod dsp;
mod editor;
mod morph;
mod nodes;
mod processor;
mod topology;

#[derive(Clone)]
pub struct TopologyConfig {
    pub destinations: Vec<String>,
}

pub struct SharedTopologyState {
    pub dirty: AtomicBool,
    pub config: Mutex<TopologyConfig>,
}

use morph::MorphParams;
use processor::Processor;

#[derive(Params)]
pub struct DtcwptMorphParams {
    #[id = "mag"]
    pub mag: FloatParam,

    #[id = "phase"]
    pub phase: FloatParam,

    #[id = "threshold"]
    pub threshold: FloatParam,

    #[id = "bypass_low"]
    pub bypass_low: BoolParam,

    #[id = "bypass_high"]
    pub bypass_high: BoolParam,
}

impl Default for DtcwptMorphParams {
    fn default() -> Self {
        Self {
            mag: FloatParam::new(
                "Magnitude",
                0.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1.0,
                },
            ),
            phase: FloatParam::new(
                "Phase",
                0.7,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1.0,
                },
            ),
            threshold: FloatParam::new(
                "Threshold",
                -20.0,
                FloatRange::Linear {
                    min: -60.0,
                    max: 0.0,
                },
            ),
            bypass_low: BoolParam::new("Bypass Low", true),
            bypass_high: BoolParam::new("Bypass High", false),
        }
    }
}

pub struct DtcwptMorph {
    params: Arc<DtcwptMorphParams>,
    processor: Option<Processor>,

    // Scratch buffers for f64 processing
    scratch_main_l: Vec<f64>,
    scratch_main_r: Vec<f64>,
    scratch_sc_l: Vec<f64>,
    scratch_sc_r: Vec<f64>,

    // Shared topology state
    topology_state: Arc<SharedTopologyState>,

    // Latency reporting
    current_latency: u32,

    // Runtime config
    sample_rate: f32,
    max_block_size: u32,
}

impl Default for DtcwptMorph {
    fn default() -> Self {
        let initial_destinations = vec![
            "H".to_string(),
            "LH".to_string(),
            "LLH".to_string(),
            "LLLH".to_string(),
            "LLLLH".to_string(),
            "LLLLLH".to_string(),
            "LLLLLL".to_string(),
        ];

        Self {
            params: Arc::new(DtcwptMorphParams::default()),
            processor: None,
            scratch_main_l: Vec::new(),
            scratch_main_r: Vec::new(),
            scratch_sc_l: Vec::new(),
            scratch_sc_r: Vec::new(),
            topology_state: Arc::new(SharedTopologyState {
                dirty: AtomicBool::new(false),
                config: Mutex::new(TopologyConfig {
                    destinations: initial_destinations,
                }),
            }),
            current_latency: 0,
            sample_rate: 44100.0,
            max_block_size: 512,
        }
    }
}

impl Plugin for DtcwptMorph {
    const NAME: &'static str = "DT-CWPT Morph";
    const VENDOR: &'static str = "LTSU";
    const URL: &'static str = "https://github.com/ltsu/dtcwpt-morph";
    const EMAIL: &'static str = "nalum@example.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            aux_input_ports: &[NonZeroU32::new(2).unwrap()],
            ..AudioIOLayout::const_default()
        },
    ];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            editor::default_state(),
            self.topology_state.clone(),
            self.sample_rate,
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;
        self.max_block_size = buffer_config.max_buffer_size;

        let destinations = {
            let config = self.topology_state.config.lock().unwrap();
            config.destinations.clone()
        };

        self.processor = Some(Processor::new(
            self.sample_rate as f64,
            self.max_block_size as usize,
            &destinations,
            2, // Stereo
        ));

        // Allocate scratch buffers
        let max_size = buffer_config.max_buffer_size as usize;
        self.scratch_main_l = vec![0.0; max_size];
        self.scratch_main_r = vec![0.0; max_size];
        self.scratch_sc_l = vec![0.0; max_size];
        self.scratch_sc_r = vec![0.0; max_size];

        // Report latency to host
        if let Some(ref proc) = self.processor {
            self.current_latency = proc.latency() as u32;
            context.set_latency_samples(self.current_latency);
        }

        true
    }

    fn reset(&mut self) {
        if let Some(ref mut proc) = self.processor {
            proc.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Check for topology changes
        if self.topology_state.dirty.swap(false, Ordering::Acquire) {
            let destinations = {
                let config = self.topology_state.config.lock().unwrap();
                config.destinations.clone()
            };

            self.processor = Some(Processor::new(
                self.sample_rate as f64,
                self.max_block_size as usize,
                &destinations,
                2,
            ));

            if let Some(ref proc) = self.processor {
                let new_latency = proc.latency() as u32;
                if new_latency != self.current_latency {
                    self.current_latency = new_latency;
                    context.set_latency_samples(new_latency);
                }
            }
        }

        let Some(ref mut proc) = self.processor else {
            return ProcessStatus::Normal;
        };

        let num_samples = buffer.samples();
        
        // Get sidechain input (if available)
        let has_sidechain = !aux.inputs.is_empty() && !aux.inputs[0].is_empty();

        // Get smoothed parameter values
        let mag = self.params.mag.smoothed.next() as f64;
        let phase = self.params.phase.smoothed.next() as f64;
        let threshold_db = self.params.threshold.smoothed.next() as f64;
        let threshold_linear = 10.0f64.powf(threshold_db / 20.0);
        let bypass_low = self.params.bypass_low.value();
        let bypass_high = self.params.bypass_high.value();

        let morph_params = MorphParams {
            mag,
            phase,
            threshold: threshold_linear,
            bypass_low,
            bypass_high,
        };

        // Convert f32 -> f64 for processing
        {
            let main_channels = buffer.as_slice();
            for (i, sample) in main_channels[0].iter().enumerate() {
                self.scratch_main_l[i] = *sample as f64;
            }
            for (i, sample) in main_channels[1].iter().enumerate() {
                self.scratch_main_r[i] = *sample as f64;
            }
        }

        // Get sidechain data
        if has_sidechain {
            let sc = aux.inputs[0].as_slice();
            for (i, sample) in sc[0].iter().enumerate() {
                self.scratch_sc_l[i] = *sample as f64;
            }
            for (i, sample) in sc[1].iter().enumerate() {
                self.scratch_sc_r[i] = *sample as f64;
            }
        } else {
            // If no sidechain, use main input as sidechain (passthrough mode)
            self.scratch_sc_l[..num_samples].copy_from_slice(&self.scratch_main_l[..num_samples]);
            self.scratch_sc_r[..num_samples].copy_from_slice(&self.scratch_main_r[..num_samples]);
        }

        // Process!
        proc.process_stereo(
            &mut self.scratch_main_l[..num_samples],
            &mut self.scratch_main_r[..num_samples],
            &self.scratch_sc_l[..num_samples],
            &self.scratch_sc_r[..num_samples],
            &morph_params,
        );

        // Convert f64 -> f32 for output
        {
            let main_channels = buffer.as_slice();
            for (i, sample) in main_channels[0].iter_mut().enumerate() {
                let out = self.scratch_main_l[i] as f32;
                *sample = out;
            }
            for (i, sample) in main_channels[1].iter_mut().enumerate() {
                let out = self.scratch_main_r[i] as f32;
                *sample = out;
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for DtcwptMorph {
    const CLAP_ID: &'static str = "com.ltsu.dtcwpt-morph";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("DT-CWPT based audio morphing effect");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for DtcwptMorph {
    const VST3_CLASS_ID: [u8; 16] = *b"DTCWPTMorphLTSU!";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
    ];
}

nih_export_clap!(DtcwptMorph);
nih_export_vst3!(DtcwptMorph);
