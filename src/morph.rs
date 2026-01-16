//! Morphing operations for DT-CWPT subbands.
//!
//! Implements magnitude interpolation and phase mixing (N-Lerp).

/// Apply morphing to complex subband buffers in-place.
///
/// # Algorithm
/// 1. Magnitude: Linear interpolation between A and B
/// 2. Phase: N-Lerp (normalized lerp) of unit vectors
///
/// # Arguments
/// * `a_re`, `a_im` - Main signal (modified in-place as output)
/// * `b_re`, `b_im` - SideChain signal (read-only)
/// * `mag_ratio` - Magnitude blend (0.0 = Main, 1.0 = SideChain)
/// * `phase_ratio` - Phase blend (0.0 = Main, 1.0 = SideChain)
/// * `threshold` - Linear amplitude threshold for phase suppression
#[inline]
pub fn morph_buffer(
    a_re: &mut [f64],
    a_im: &mut [f64],
    b_re: &[f64],
    b_im: &[f64],
    mag_ratio: f64,
    phase_ratio: f64,
    threshold: f64,
) {
    debug_assert_eq!(a_re.len(), a_im.len());
    debug_assert_eq!(a_re.len(), b_re.len());
    debug_assert_eq!(a_re.len(), b_im.len());

    const EPS: f64 = 1e-37;

    for i in 0..a_re.len() {
        // 1. Calculate magnitudes
        let mag_a = (a_re[i] * a_re[i] + a_im[i] * a_im[i]).sqrt();
        let mag_b = (b_re[i] * b_re[i] + b_im[i] * b_im[i]).sqrt();

        // Adjust phase coefficient based on sidechain signal level
        let current_phase_coef = if mag_b < threshold {
            phase_ratio * mag_b * 0.9 / threshold
        } else {
            phase_ratio
        };

        // 2. Linear interpolation of magnitude
        let target_mag = mag_a * (1.0 - mag_ratio) + mag_b * mag_ratio;

        // 3. Unit vectors
        let scale_a = 1.0 / (mag_a + EPS);
        let ua_re = a_re[i] * scale_a;
        let ua_im = a_im[i] * scale_a;

        let scale_b = 1.0 / (mag_b + EPS);
        let ub_re = b_re[i] * scale_b;
        let ub_im = b_im[i] * scale_b;

        // 4. N-Lerp: weighted sum then normalize
        let mix_re = ua_re * (1.0 - current_phase_coef) + ub_re * current_phase_coef;
        let mix_im = ua_im * (1.0 - current_phase_coef) + ub_im * current_phase_coef;

        let mix_mag = (mix_re * mix_re + mix_im * mix_im).sqrt();
        let scale_mix = 1.0 / (mix_mag + EPS);

        let unit_mix_re = mix_re * scale_mix;
        let unit_mix_im = mix_im * scale_mix;

        // 5. Final output: target_mag * unit_phase
        a_re[i] = target_mag * unit_mix_re;
        a_im[i] = target_mag * unit_mix_im;
    }
}

/// Morphing parameters.
#[derive(Clone, Copy, Debug)]
pub struct MorphParams {
    /// Magnitude interpolation ratio (0.0 = Main, 1.0 = SideChain)
    pub mag: f64,
    /// Phase mixing ratio (0.0 = Main, 1.0 = SideChain)
    pub phase: f64,
    /// Linear amplitude threshold for phase processing
    pub threshold: f64,
    /// Bypass morphing on the lowest frequency band (pure L path)
    pub bypass_low: bool,
    /// Bypass morphing on the highest frequency band (pure H path)
    pub bypass_high: bool,
}

impl Default for MorphParams {
    fn default() -> Self {
        Self {
            mag: 0.01,
            phase: 1.0,
            threshold: 0.01,
            bypass_low: false,
            bypass_high: false,
        }
    }
}

impl MorphParams {
    /// Create MorphParams with threshold specified in decibels.
    pub fn with_threshold_db(mag: f64, phase: f64, threshold_db: f64, bypass_low: bool, bypass_high: bool) -> Self {
        Self {
            mag,
            phase,
            threshold: 10.0f64.powf(threshold_db / 20.0),
            bypass_low,
            bypass_high,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morph_passthrough() {
        let mut a_re = vec![1.0, 2.0, 3.0];
        let mut a_im = vec![0.5, 1.0, 1.5];
        let b_re = vec![0.0, 0.0, 0.0];
        let b_im = vec![0.0, 0.0, 0.0];

        // mag=0, phase=0 should keep A unchanged
        morph_buffer(&mut a_re, &mut a_im, &b_re, &b_im, 0.0, 0.0, 0.01);

        assert!((a_re[0] - 1.0).abs() < 1e-10);
        assert!((a_re[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_threshold_db_conversion() {
        let params = MorphParams::with_threshold_db(0.5, 0.5, -20.0);
        assert!((params.threshold - 0.1).abs() < 1e-10);
    }
}
