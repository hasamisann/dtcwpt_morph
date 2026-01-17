// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 LTSU

//! Main DT-CWPT processor engine.
//!
//! Manages the full processing pipeline:
//! Analysis -> Morphing -> Delay Compensation -> Synthesis

use crate::constants::*;
use crate::dsp::DelayBuffer;
use crate::morph::{morph_buffer, MorphParams};
use crate::nodes::{AnalysisNode, SynthesisNode};
use crate::topology::{TopologyPlanner, MAX_TREE_SIZE};

/// Main DT-CWPT processing engine.
pub struct Processor {
    /// Number of audio channels
    channel_num: usize,
    /// Sample rate
    sample_rate: f64,
    /// Topology planner
    topology: TopologyPlanner,
    /// Maximum computed delay
    max_delay: usize,

    // Analysis nodes (per channel, per tree)
    main_analysis_re: Vec<Vec<AnalysisNode>>,
    main_analysis_im: Vec<Vec<AnalysisNode>>,
    sc_analysis_re: Vec<Vec<AnalysisNode>>,
    sc_analysis_im: Vec<Vec<AnalysisNode>>,
    analysis_node_ids: Vec<usize>,

    // Synthesis nodes (per channel)
    synthesis_re: Vec<Vec<SynthesisNode>>,
    synthesis_im: Vec<Vec<SynthesisNode>>,
    synthesis_node_ids: Vec<usize>,

    // Delay buffers (per channel, per destination)
    main_delay_re: Vec<Vec<DelayBuffer>>,
    main_delay_im: Vec<Vec<DelayBuffer>>,

    // Work buffers for analysis (shared scratch space)
    work_buffer: Vec<f64>,
    active_flags: Vec<bool>,

    // Result buffers (per destination node)
    main_results_re: Vec<Option<Vec<f64>>>,
    main_results_im: Vec<Option<Vec<f64>>>,
    sc_results_re: Vec<Option<Vec<f64>>>,
    sc_results_im: Vec<Option<Vec<f64>>>,

    // Cursors for result buffers
    main_cursors_re: Vec<usize>,
    main_cursors_im: Vec<usize>,
    sc_cursors_re: Vec<usize>,
    sc_cursors_im: Vec<usize>,
}

impl Processor {
    /// Create a new processor.
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate
    /// * `max_block_size` - Maximum block size from host
    /// * `destinations` - DT-CWPT destination paths
    /// * `channel_num` - Number of audio channels
    pub fn new(
        sample_rate: f64,
        max_block_size: usize,
        destinations: &[String],
        channel_num: usize,
    ) -> Self {
        let topology = TopologyPlanner::new(destinations);

        // Initialize analysis nodes
        let mut main_analysis_re = vec![Vec::new(); channel_num];
        let mut main_analysis_im = vec![Vec::new(); channel_num];
        let mut analysis_node_ids = Vec::new();

        // Root node (level 1, uses CDF filters)
        for ch in 0..channel_num {
            main_analysis_re[ch].push(AnalysisNode::new(
                &CDF_H0R,
                CDF_DELAY_RE,
                &CDF_H1R,
                CDF_DELAY_RE,
            ));
            main_analysis_im[ch].push(AnalysisNode::new(
                &CDF_H0I,
                CDF_DELAY_IM,
                &CDF_H1I,
                CDF_DELAY_IM,
            ));
        }
        analysis_node_ids.push(1);

        // Non-root nodes
        for &idx in &topology.analysis_order {
            if idx == 1 {
                continue;
            }
            analysis_node_ids.push(idx);

            let is_high_child = (idx % 2) == 1;
            for ch in 0..channel_num {
                if is_high_child {
                    // Packet filters (same for both trees)
                    main_analysis_re[ch].push(AnalysisNode::new(
                        &PACKET_H0,
                        PACKET_DELAY,
                        &PACKET_H1,
                        PACKET_DELAY,
                    ));
                    main_analysis_im[ch].push(AnalysisNode::new(
                        &PACKET_H0,
                        PACKET_DELAY,
                        &PACKET_H1,
                        PACKET_DELAY,
                    ));
                } else {
                    // Q-Shift filters
                    main_analysis_re[ch].push(AnalysisNode::new(
                        &QSHIFT14_H0R,
                        QSHIFT14_DELAY_RE,
                        &QSHIFT14_H1R,
                        QSHIFT14_DELAY_RE,
                    ));
                    main_analysis_im[ch].push(AnalysisNode::new(
                        &QSHIFT14_H0I,
                        QSHIFT14_DELAY_IM,
                        &QSHIFT14_H1I,
                        QSHIFT14_DELAY_IM,
                    ));
                }
            }
        }

        // Clone for sidechain
        let sc_analysis_re = main_analysis_re.clone();
        let sc_analysis_im = main_analysis_im.clone();

        // Initialize synthesis nodes
        let mut synthesis_re = vec![Vec::new(); channel_num];
        let mut synthesis_im = vec![Vec::new(); channel_num];
        let mut synthesis_node_ids = Vec::new();

        for &idx in &topology.synthesis_order {
            synthesis_node_ids.push(idx);

            if idx == 1 {
                // Root: CDF filters
                for ch in 0..channel_num {
                    synthesis_re[ch].push(SynthesisNode::new(
                        &CDF_G0R,
                        CDF_DELAY_RE,
                        &CDF_G1R,
                        CDF_DELAY_RE,
                    ));
                    synthesis_im[ch].push(SynthesisNode::new(
                        &CDF_G0I,
                        CDF_DELAY_IM,
                        &CDF_G1I,
                        CDF_DELAY_IM,
                    ));
                }
            } else {
                let is_high_child = (idx % 2) == 1;
                for ch in 0..channel_num {
                    if is_high_child {
                        // Packet
                        synthesis_re[ch].push(SynthesisNode::new(
                            &PACKET_G0,
                            PACKET_DELAY,
                            &PACKET_G1,
                            PACKET_DELAY,
                        ));
                        synthesis_im[ch].push(SynthesisNode::new(
                            &PACKET_G0,
                            PACKET_DELAY,
                            &PACKET_G1,
                            PACKET_DELAY,
                        ));
                    } else {
                        // Q-Shift
                        synthesis_re[ch].push(SynthesisNode::new(
                            &QSHIFT14_G0R,
                            QSHIFT14_DELAY_RE,
                            &QSHIFT14_G1R,
                            QSHIFT14_DELAY_RE,
                        ));
                        synthesis_im[ch].push(SynthesisNode::new(
                            &QSHIFT14_G0I,
                            QSHIFT14_DELAY_IM,
                            &QSHIFT14_G1I,
                            QSHIFT14_DELAY_IM,
                        ));
                    }
                }
            }
        }

        // Calculate delays
        let mut max_delay = 0usize;
        let mut path_delays_re = vec![0usize; MAX_TREE_SIZE];
        let mut path_delays_im = vec![0usize; MAX_TREE_SIZE];

        // Build a map of node_id -> dense_index for synthesis nodes
        let synth_id_map: std::collections::HashMap<usize, usize> = synthesis_node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        for &dest in &topology.destinations {
            let d_re = calculate_delay(dest, &synthesis_re[0], &synthesis_node_ids, &synth_id_map, false);
            let d_im = calculate_delay(dest, &synthesis_im[0], &synthesis_node_ids, &synth_id_map, true);
            path_delays_re[dest] = d_re;
            path_delays_im[dest] = d_im;
            max_delay = max_delay.max(d_re).max(d_im);
        }

        // Initialize delay buffers
        let mut main_delay_re = vec![Vec::new(); channel_num];
        let mut main_delay_im = vec![Vec::new(); channel_num];

        for &dest in &topology.destinations {
            let depth = TopologyPlanner::node_depth(dest);
            let rate_factor = 1 << depth;

            let diff_re = (max_delay - path_delays_re[dest]) / rate_factor;
            let diff_im = (max_delay - path_delays_im[dest]) / rate_factor;

            for ch in 0..channel_num {
                main_delay_re[ch].push(DelayBuffer::new(diff_re, max_block_size));
                main_delay_im[ch].push(DelayBuffer::new(diff_im, max_block_size));
            }
        }

        // Initialize result buffers
        let mut main_results_re: Vec<Option<Vec<f64>>> = vec![None; MAX_TREE_SIZE];
        let mut main_results_im: Vec<Option<Vec<f64>>> = vec![None; MAX_TREE_SIZE];
        let mut sc_results_re: Vec<Option<Vec<f64>>> = vec![None; MAX_TREE_SIZE];
        let mut sc_results_im: Vec<Option<Vec<f64>>> = vec![None; MAX_TREE_SIZE];

        // Allocate for analysis order nodes
        for &target in &topology.analysis_order {
            let depth = TopologyPlanner::node_depth(target);
            let output_size = max_block_size >> depth;
            main_results_re[target] = Some(vec![0.0; output_size]);
            main_results_im[target] = Some(vec![0.0; output_size]);
            sc_results_re[target] = Some(vec![0.0; output_size]);
            sc_results_im[target] = Some(vec![0.0; output_size]);
        }

        // Allocate for destination nodes
        for &target in &topology.destinations {
            let depth = TopologyPlanner::node_depth(target);
            let output_size = max_block_size >> depth;
            main_results_re[target] = Some(vec![0.0; output_size]);
            main_results_im[target] = Some(vec![0.0; output_size]);
            sc_results_re[target] = Some(vec![0.0; output_size]);
            sc_results_im[target] = Some(vec![0.0; output_size]);
        }

        Self {
            channel_num,
            sample_rate,
            topology,
            max_delay,
            main_analysis_re,
            main_analysis_im,
            sc_analysis_re,
            sc_analysis_im,
            analysis_node_ids,
            synthesis_re,
            synthesis_im,
            synthesis_node_ids,
            main_delay_re,
            main_delay_im,
            work_buffer: vec![0.0; MAX_TREE_SIZE],
            active_flags: vec![false; MAX_TREE_SIZE],
            main_results_re,
            main_results_im,
            sc_results_re,
            sc_results_im,
            main_cursors_re: vec![0; MAX_TREE_SIZE],
            main_cursors_im: vec![0; MAX_TREE_SIZE],
            sc_cursors_re: vec![0; MAX_TREE_SIZE],
            sc_cursors_im: vec![0; MAX_TREE_SIZE],
        }
    }

    /// Get the total latency in samples.
    pub fn latency(&self) -> usize {
        self.max_delay
    }

    /// Get the topology planner.
    pub fn topology(&self) -> &TopologyPlanner {
        &self.topology
    }

    /// Process a stereo block.
    ///
    /// # Arguments
    /// * `main_l`, `main_r` - Main input (modified in-place)
    /// * `sc_l`, `sc_r` - Sidechain input (read-only)
    /// * `params` - Morphing parameters
    pub fn process_stereo(
        &mut self,
        main_l: &mut [f64],
        main_r: &mut [f64],
        sc_l: &[f64],
        sc_r: &[f64],
        params: &MorphParams,
    ) {
        let main_channels = [main_l, main_r];
        let sc_channels = [sc_l, sc_r];
        
        // Process each channel
        for (ch, (main, sc)) in main_channels.into_iter().zip(sc_channels.into_iter()).enumerate() {
            self.process_channel(ch, main, sc, params);
        }
    }

    /// Process a single channel.
    fn process_channel(
        &mut self,
        ch: usize,
        main: &mut [f64],
        sc: &[f64],
        params: &MorphParams,
    ) {
        let buffer_size = main.len();

        // Reset cursors
        for &target in self.topology.analysis_order.iter().chain(self.topology.destinations.iter()) {
            self.main_cursors_re[target] = 0;
            self.main_cursors_im[target] = 0;
            self.sc_cursors_re[target] = 0;
            self.sc_cursors_im[target] = 0;
        }

        // ============ Analysis ============
        self.run_analysis(ch, main, true, true);  // Main Re
        self.run_analysis(ch, main, true, false); // Main Im
        self.run_analysis(ch, sc, false, true);   // SC Re
        self.run_analysis(ch, sc, false, false);  // SC Im

        // ============ Morphing ============
        // Process all destination bands, respecting bypass flags
        for &dest in &self.topology.destinations {
            // Check if this band should be bypassed
            let is_pure_l = dest.is_power_of_two(); // Pure L path: 2, 4, 8, 16, ...
            let is_pure_h = dest == 3 || (dest > 1 && (dest & (dest + 1)) == 0 && dest.count_ones() == (dest.trailing_ones())); // Pure H path: just "H" = 3
            
            // Skip morphing if bypass is enabled for this band
            if (is_pure_l && params.bypass_low) || (dest == 3 && params.bypass_high) {
                continue;
            }

            let re_len = self.main_cursors_re[dest];
            let im_len = self.main_cursors_im[dest];

            if let (Some(main_re), Some(main_im), Some(sc_re), Some(sc_im)) = (
                self.main_results_re[dest].as_mut(),
                self.main_results_im[dest].as_mut(),
                self.sc_results_re[dest].as_ref(),
                self.sc_results_im[dest].as_ref(),
            ) {
                morph_buffer(
                    &mut main_re[..re_len],
                    &mut main_im[..im_len],
                    &sc_re[..re_len],
                    &sc_im[..im_len],
                    params.mag,
                    params.phase,
                    params.threshold,
                );
            }
        }

        // ============ Delay Compensation ============
        for (i, &dest) in self.topology.destinations.iter().enumerate() {
            let len_re = self.main_cursors_re[dest];
            let len_im = self.main_cursors_im[dest];

            if let Some(buf) = self.main_results_re[dest].as_mut() {
                self.main_delay_re[ch][i].process_inplace(&mut buf[..len_re]);
            }
            if let Some(buf) = self.main_results_im[dest].as_mut() {
                self.main_delay_im[ch][i].process_inplace(&mut buf[..len_im]);
            }
        }

        // ============ Synthesis ============
        self.run_synthesis(ch, true);  // Re
        self.run_synthesis(ch, false); // Im

        // ============ Output ============
        // Average Re and Im trees
        if let (Some(root_re), Some(root_im)) = (
            self.main_results_re[1].as_ref(),
            self.main_results_im[1].as_ref(),
        ) {
            for i in 0..buffer_size {
                main[i] = (root_re[i] + root_im[i]) * 0.5;
            }
        }
    }

    /// Run analysis for one tree.
    fn run_analysis(&mut self, ch: usize, input: &[f64], is_main: bool, is_re: bool) {
        let nodes = if is_main {
            if is_re { &mut self.main_analysis_re[ch] } else { &mut self.main_analysis_im[ch] }
        } else {
            if is_re { &mut self.sc_analysis_re[ch] } else { &mut self.sc_analysis_im[ch] }
        };

        let results = if is_main {
            if is_re { &mut self.main_results_re } else { &mut self.main_results_im }
        } else {
            if is_re { &mut self.sc_results_re } else { &mut self.sc_results_im }
        };

        let cursors = if is_main {
            if is_re { &mut self.main_cursors_re } else { &mut self.main_cursors_im }
        } else {
            if is_re { &mut self.sc_cursors_re } else { &mut self.sc_cursors_im }
        };

        // Process sample by sample
        for &x in input {
            self.work_buffer[1] = x;
            self.active_flags.fill(false);
            self.active_flags[1] = true;

            for (node, &node_id) in nodes.iter_mut().zip(self.analysis_node_ids.iter()) {
                if !self.active_flags[node_id] {
                    continue;
                }

                let left_idx = node_id << 1;
                let right_idx = (node_id << 1) | 1;

                if let Some((low, high)) = node.process_sample(self.work_buffer[node_id]) {
                    self.work_buffer[left_idx] = low;
                    self.work_buffer[right_idx] = high;
                    self.active_flags[left_idx] = true;
                    self.active_flags[right_idx] = true;
                }
            }

            // Save results to destination buffers
            for &dest in &self.topology.destinations {
                if self.active_flags[dest] {
                    if let Some(buf) = results[dest].as_mut() {
                        let cursor = cursors[dest];
                        buf[cursor] = self.work_buffer[dest];
                        cursors[dest] = cursor + 1;
                    }
                }
            }
        }
    }

    /// Run synthesis for one tree.
    fn run_synthesis(&mut self, ch: usize, is_re: bool) {
        let nodes = if is_re { &mut self.synthesis_re[ch] } else { &mut self.synthesis_im[ch] };
        let results = if is_re { &mut self.main_results_re } else { &mut self.main_results_im };
        let cursors = if is_re { &mut self.main_cursors_re } else { &mut self.main_cursors_im };

        for (node, &parent_idx) in nodes.iter_mut().zip(self.synthesis_node_ids.iter()) {
            let left_idx = parent_idx << 1;
            let right_idx = (parent_idx << 1) | 1;

            let num_samples = cursors[left_idx];
            let write_cursor = cursors[parent_idx];

            // Copy data to avoid borrow conflicts
            let low_data: Option<Vec<f64>> = results[left_idx].as_ref().map(|b| b[..num_samples].to_vec());
            let high_data: Option<Vec<f64>> = results[right_idx].as_ref().map(|b| b[..num_samples].to_vec());

            if let (Some(low_buf), Some(high_buf), Some(out_buf)) = (
                low_data.as_deref(),
                high_data.as_deref(),
                results[parent_idx].as_mut(),
            ) {
                node.synthesize_block(low_buf, high_buf, out_buf, write_cursor, num_samples);
                cursors[parent_idx] = write_cursor + num_samples * 2;
            }
        }
    }


    /// Reset all internal state.
    pub fn reset(&mut self) {
        for ch in 0..self.channel_num {
            for node in &mut self.main_analysis_re[ch] {
                node.reset();
            }
            for node in &mut self.main_analysis_im[ch] {
                node.reset();
            }
            for node in &mut self.sc_analysis_re[ch] {
                node.reset();
            }
            for node in &mut self.sc_analysis_im[ch] {
                node.reset();
            }
            for node in &mut self.synthesis_re[ch] {
                node.reset();
            }
            for node in &mut self.synthesis_im[ch] {
                node.reset();
            }
            for buf in &mut self.main_delay_re[ch] {
                buf.reset();
            }
            for buf in &mut self.main_delay_im[ch] {
                buf.reset();
            }
        }

        self.work_buffer.fill(0.0);
        self.active_flags.fill(false);

        for buf in self.main_results_re.iter_mut().chain(self.main_results_im.iter_mut())
            .chain(self.sc_results_re.iter_mut()).chain(self.sc_results_im.iter_mut())
        {
            if let Some(b) = buf {
                b.fill(0.0);
            }
        }
    }
}

/// Calculate delay for a path through synthesis nodes.
fn calculate_delay(
    target_idx: usize,
    nodes: &[SynthesisNode],
    node_ids: &[usize],
    id_map: &std::collections::HashMap<usize, usize>,
    _is_im: bool,
) -> usize {
    let mut total_delay = 0;
    let mut coef = 1;
    let num_bits = (usize::BITS - target_idx.leading_zeros()) as usize;
    let mut current_node = 1;

    for bit_pos in (0..num_bits - 1).rev() {
        let direction = (target_idx >> bit_pos) & 1;

        let delay = if let Some(&dense_idx) = id_map.get(&current_node) {
            let node = &nodes[dense_idx];
            if direction == 0 {
                node.delay_low()
            } else {
                node.delay_high()
            }
        } else {
            0
        };

        if direction == 0 {
            current_node <<= 1;
        } else {
            current_node = (current_node << 1) | 1;
        }

        total_delay += delay * coef;
        coef <<= 1;
    }

    total_delay
}
