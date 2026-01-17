// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 LTSU

//! Wavelet tree nodes for analysis and synthesis.
//!
//! - `AnalysisNode`: Downsampling + filtering (分解)
//! - `SynthesisNode`: Upsampling + filtering (合成)

use crate::dsp::StatefulFilter;

// ==============================================================================
// AnalysisNode: Decomposition node
// ==============================================================================

/// A node in the analysis (decomposition) tree.
///
/// Applies lowpass and highpass filters, then downsamples by 2.
#[derive(Clone)]
pub struct AnalysisNode {
    filter_low: StatefulFilter,
    filter_high: StatefulFilter,
    /// Parity flag for downsampling (false = output, true = skip)
    parity: bool,
}

impl AnalysisNode {
    /// Create a new analysis node with the given filter kernels.
    pub fn new(kernel_low: &[f64], delay_low: usize, kernel_high: &[f64], delay_high: usize) -> Self {
        Self {
            filter_low: StatefulFilter::new(kernel_low, delay_low),
            filter_high: StatefulFilter::new(kernel_high, delay_high),
            parity: false,
        }
    }

    /// Process one input sample.
    ///
    /// # Returns
    /// `Some((low, high))` if output is produced (even samples), `None` if skipped (odd samples).
    #[inline]
    pub fn process_sample(&mut self, x: f64) -> Option<(f64, f64)> {
        self.filter_low.push_sample(x);
        self.filter_high.push_sample(x);

        if self.parity {
            self.parity = false;
            None
        } else {
            let low = self.filter_low.filter();
            let high = self.filter_high.filter();
            self.parity = true;
            Some((low, high))
        }
    }

    /// Reset the node state.
    pub fn reset(&mut self) {
        self.filter_low.reset();
        self.filter_high.reset();
        self.parity = false;
    }

    /// Get the delay of the lowpass filter.
    pub fn delay_low(&self) -> usize {
        self.filter_low.delay
    }

    /// Get the delay of the highpass filter.
    pub fn delay_high(&self) -> usize {
        self.filter_high.delay
    }
}

// ==============================================================================
// SynthesisNode: Reconstruction node
// ==============================================================================

/// A node in the synthesis (reconstruction) tree.
///
/// Upsamples by 2 (zero insertion) and applies lowpass/highpass filters.
#[derive(Clone)]
pub struct SynthesisNode {
    filter_low: StatefulFilter,
    filter_high: StatefulFilter,
}

impl SynthesisNode {
    /// Create a new synthesis node with the given filter kernels.
    pub fn new(kernel_low: &[f64], delay_low: usize, kernel_high: &[f64], delay_high: usize) -> Self {
        Self {
            filter_low: StatefulFilter::new(kernel_low, delay_low),
            filter_high: StatefulFilter::new(kernel_high, delay_high),
        }
    }

    /// Synthesize a block of samples.
    ///
    /// # Arguments
    /// * `low_buf` - Low frequency subband input
    /// * `high_buf` - High frequency subband input
    /// * `out_buf` - Output buffer (length = 2 * input length)
    /// * `out_offset` - Starting offset in output buffer
    /// * `num_samples` - Number of input samples to process
    pub fn synthesize_block(
        &mut self,
        low_buf: &[f64],
        high_buf: &[f64],
        out_buf: &mut [f64],
        out_offset: usize,
        num_samples: usize,
    ) {
        for i in 0..num_samples {
            // Even sample (actual input)
            self.filter_low.push_sample(low_buf[i]);
            self.filter_high.push_sample(high_buf[i]);
            out_buf[out_offset + 2 * i] = self.filter_low.filter() + self.filter_high.filter();

            // Odd sample (zero padding for upsampling)
            self.filter_low.push_sample(0.0);
            self.filter_high.push_sample(0.0);
            out_buf[out_offset + 2 * i + 1] = self.filter_low.filter() + self.filter_high.filter();
        }
    }

    /// Reset the node state.
    pub fn reset(&mut self) {
        self.filter_low.reset();
        self.filter_high.reset();
    }

    /// Get the delay of the lowpass filter.
    pub fn delay_low(&self) -> usize {
        self.filter_low.delay
    }

    /// Get the delay of the highpass filter.
    pub fn delay_high(&self) -> usize {
        self.filter_high.delay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_node_downsampling() {
        // Simple identity-ish kernel for testing
        let kernel = vec![1.0];
        let mut node = AnalysisNode::new(&kernel, 0, &kernel, 0);

        // First sample: should produce output
        let result = node.process_sample(1.0);
        assert!(result.is_some());

        // Second sample: should be skipped
        let result = node.process_sample(2.0);
        assert!(result.is_none());

        // Third sample: should produce output
        let result = node.process_sample(3.0);
        assert!(result.is_some());
    }
}
