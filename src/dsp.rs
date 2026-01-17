// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 LTSU

//! DSP building blocks for DT-CWPT processing.
//!
//! Contains:
//! - `StatefulFilter`: Ring buffer-based FIR filter
//! - `DelayBuffer`: Sample delay compensation buffer


// ==============================================================================
// StatefulFilter: Ring buffer FIR filter
// ==============================================================================

/// A stateful FIR filter using a mirrored ring buffer for efficient convolution.
///
/// The mirror buffer technique eliminates modulo operations during convolution
/// by maintaining two copies of the data.
#[derive(Clone)]
pub struct StatefulFilter {
    /// Filter kernel coefficients
    kernel: Vec<f64>,
    /// Mirrored ring buffer (size = 2 * buffer_size)
    buffer: Vec<f64>,
    /// Buffer size (kernel.len() + 1)
    size: usize,
    /// Current write position in buffer
    cursor: usize,
    /// Filter delay in samples
    pub delay: usize,
}

impl StatefulFilter {
    /// Create a new stateful filter with the given kernel and delay.
    pub fn new(kernel: &[f64], delay: usize) -> Self {
        let size = kernel.len() + 1;
        Self {
            kernel: kernel.to_vec(),
            buffer: vec![0.0; size * 2],
            size,
            cursor: 0,
            delay,
        }
    }

    /// Push a single sample into the buffer.
    #[inline]
    pub fn push_sample(&mut self, x: f64) {
        self.buffer[self.cursor] = x;
        self.buffer[self.cursor + self.size] = x;
        self.cursor += 1;
        if self.cursor >= self.size {
            self.cursor = 0;
        }
    }

    /// Compute the filter output (convolution with kernel).
    ///
    /// This is a hot loop - candidate for SIMD optimization.
    #[inline]
    pub fn filter(&self) -> f64 {
        let mut idx = if self.cursor == 0 {
            self.size - 1 + self.size
        } else {
            self.cursor - 1 + self.size
        };

        let mut acc = 0.0;
        for &k in &self.kernel {
            // SAFETY: idx is always in bounds due to mirror buffer
            acc += k * self.buffer[idx];
            idx -= 1;
        }
        acc
    }

    /// Reset the filter state to zeros.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.cursor = 0;
    }
}

// ==============================================================================
// DelayBuffer: Ring buffer for delay compensation
// ==============================================================================

/// A delay buffer using a mirrored ring buffer for block-based delay processing.
///
/// Used to align subband signals with different path delays.
#[derive(Clone)]
pub struct DelayBuffer {
    /// Mirrored ring buffer
    buffer: Vec<f64>,
    /// Buffer size (power of 2)
    size: usize,
    /// Current write cursor
    write_cursor: usize,
    /// Delay in samples
    delay: usize,
}

impl DelayBuffer {
    /// Create a new delay buffer.
    ///
    /// # Arguments
    /// * `delay_samples` - Number of samples to delay
    /// * `block_size` - Maximum block size to process
    pub fn new(delay_samples: usize, block_size: usize) -> Self {
        // Calculate buffer size as next power of 2
        let min_size = block_size * 2 + delay_samples;
        let size = min_size.next_power_of_two();

        Self {
            buffer: vec![0.0; size * 2], // Mirrored buffer
            size,
            write_cursor: 0,
            delay: delay_samples,
        }
    }

    /// Process an input block, writing delayed output to output_buffer.
    ///
    /// # Arguments
    /// * `input` - Input samples (read-only)
    /// * `output` - Output buffer (will be overwritten with delayed samples)
    pub fn process(&mut self, input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());

        // Zero delay: just copy
        if self.delay == 0 {
            output.copy_from_slice(input);
            return;
        }

        let n = input.len();
        let end_pos = self.write_cursor + n;

        // Write to both halves of mirrored buffer
        if end_pos <= self.size {
            // No wrap
            self.buffer[self.write_cursor..end_pos].copy_from_slice(input);
            self.buffer[self.write_cursor + self.size..end_pos + self.size]
                .copy_from_slice(input);
        } else {
            // Wrap around
            let remain = self.size - self.write_cursor;
            self.buffer[self.write_cursor..self.size].copy_from_slice(&input[..remain]);
            self.buffer[self.write_cursor + self.size..self.size * 2]
                .copy_from_slice(&input[..remain]);
            self.buffer[..n - remain].copy_from_slice(&input[remain..]);
            self.buffer[self.size..self.size + n - remain].copy_from_slice(&input[remain..]);
        }

        // Calculate read position
        let mask = self.size - 1;
        let current_head = (self.write_cursor + n) & mask;
        let read_cursor = (current_head.wrapping_sub(n).wrapping_sub(self.delay)) & mask;

        // Update write cursor
        self.write_cursor = current_head;

        // Read from mirrored buffer (no wrap needed)
        output.copy_from_slice(&self.buffer[read_cursor..read_cursor + n]);
    }

    /// Process in-place (output overwrites input slice).
    pub fn process_inplace(&mut self, buffer: &mut [f64]) {
        // Use a temporary copy for the read phase
        // In a real implementation, we'd use a scratch buffer
        let input_copy: Vec<f64> = buffer.to_vec();
        self.process(&input_copy, buffer);
    }

    /// Reset the buffer state.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_cursor = 0;
    }

    /// Get the delay in samples.
    pub fn delay(&self) -> usize {
        self.delay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_buffer_zero_delay() {
        let mut buf = DelayBuffer::new(0, 64);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        buf.process(&input, &mut output);
        assert_eq!(input, output);
    }

    #[test]
    fn test_delay_buffer_nonzero_delay() {
        let mut buf = DelayBuffer::new(4, 64);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        
        // First block: should output zeros (delayed)
        buf.process(&input, &mut output);
        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
        
        // Second block: should output first input
        let input2 = vec![5.0, 6.0, 7.0, 8.0];
        buf.process(&input2, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn test_stateful_filter_impulse() {
        let kernel = vec![1.0, 0.5, 0.25];
        let mut filter = StatefulFilter::new(&kernel, 0);
        
        // Push impulse
        filter.push_sample(1.0);
        assert!((filter.filter() - 1.0).abs() < 1e-10);
        
        filter.push_sample(0.0);
        assert!((filter.filter() - 0.5).abs() < 1e-10);
        
        filter.push_sample(0.0);
        assert!((filter.filter() - 0.25).abs() < 1e-10);
        
        filter.push_sample(0.0);
        assert!((filter.filter() - 0.0).abs() < 1e-10);
    }
}
