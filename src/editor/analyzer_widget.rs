// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 LTSU

//! Spectrum analyzer widget for visualizing input/output frequency content.

use nih_plug_egui::egui::{self, Color32, Pos2, Rect, Stroke, Mesh, epaint::Vertex};
use crate::analyzer::{SpectrumAnalyzer, ANALYZER_NUM_BINS, ANALYZER_FFT_SIZE};

const DB_MIN: f32 = -80.0;
const DB_MAX: f32 = 0.0;
const FREQ_MIN: f32 = 15.0;

// Colors (matching plugin palette)
const INPUT_LINE: Color32 = Color32::from_rgb(150, 150, 150);
const OUTPUT_LINE: Color32 = Color32::from_rgb(115, 170, 230);

const MARGIN_RIGHT: f32 = 10.0;
const TOP_PADDING: f32 = 5.0;
const BOTTOM_PADDING: f32 = 15.0;
const TEXT_COLOR: Color32 = Color32::from_rgb(150, 150, 150);

/// Draw the spectrum analyzer widget.
pub fn draw_spectrum_analyzer(
    ui: &mut egui::Ui,
    rect: Rect,
    analyzer: &mut SpectrumAnalyzer,
    sample_rate: f32,
) {
    let painter = ui.painter().with_clip_rect(rect);
    let nyquist = sample_rate / 2.0;

let draw_rect = Rect::from_min_max(
    Pos2::new(rect.min.x - 2.0, rect.min.y + TOP_PADDING),
    Pos2::new(rect.max.x - MARGIN_RIGHT, rect.max.y - BOTTOM_PADDING),
);

    // Fill colors (using unmultiplied alpha for intuitive control)
    let input_fill = Color32::from_rgba_unmultiplied(150, 150, 150, 30); // gray
    let output_fill = Color32::from_rgba_unmultiplied(115, 170, 230, 3); // blue

    // 2. Compute Spectrum
    let (input_mags, output_mags) = analyzer.compute();

    // 3. Draw Input Spectrum
    let input_points = build_spectrum_path(input_mags, draw_rect, sample_rate, nyquist);
if !input_points.is_empty() {
    // Fill
    let mesh = create_spectrum_fill(&input_points, draw_rect.max.y, input_fill);
    painter.add(mesh);

    // Stroke (thinner)
    let stroke_points: Vec<Pos2> = input_points.clone();
    painter.add(egui::Shape::line(stroke_points, Stroke::new(1.0, INPUT_LINE)));
}

    // 4. Draw Output Spectrum
    let output_points = build_spectrum_path(output_mags, draw_rect, sample_rate, nyquist);
    if !output_points.is_empty() {
        // Fill
        let mesh = create_spectrum_fill(&output_points, draw_rect.max.y, output_fill);
        painter.add(mesh);

        // Stroke (thinner)
        painter.add(egui::Shape::line(output_points, Stroke::new(1.5, OUTPUT_LINE)));
    }

    // Frequency Labels (on TOP of spectrum)
    for &freq in &[50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0] {
        if freq < nyquist {
            let x = draw_rect.min.x + freq_to_x(freq, FREQ_MIN, nyquist, draw_rect.width());
            let text = if freq >= 1000.0 {
                format!("{:.0}k", freq / 1000.0)
            } else {
                format!("{:.0}", freq)
            };
            painter.text(
                Pos2::new(x + 2.0, draw_rect.max.y + 10.0),
                egui::Align2::LEFT_BOTTOM,
                text,
                egui::FontId::proportional(10.0),
                TEXT_COLOR,
            );
        }
    }
}

/// Convert frequency (Hz) to x-coordinate using logarithmic scale.
fn freq_to_x(freq: f32, freq_min: f32, freq_max: f32, width: f32) -> f32 {
    let log_min = freq_min.ln();
    let log_max = freq_max.ln();
    let log_freq = freq.ln();
    ((log_freq - log_min) / (log_max - log_min)) * width
}

/// Convert FFT bin index to frequency (Hz).
fn bin_to_freq(bin: usize, sample_rate: f32, fft_size: usize) -> f32 {
    (bin as f32) * sample_rate / (fft_size as f32)
}

/// Convert dB value to y-coordinate in the given rect.
fn db_to_y(db: f32, rect: Rect) -> f32 {
    let t = (db - DB_MIN) / (DB_MAX - DB_MIN);
    let t_clamped = t.clamp(0.0, 1.0);
    rect.max.y - t_clamped * rect.height()
}

/// Build a path of screen coordinates from FFT magnitude data.
fn build_spectrum_path(mags: &[f32], rect: Rect, sample_rate: f32, freq_max: f32) -> Vec<Pos2> {
    let mut points = Vec::with_capacity(ANALYZER_NUM_BINS);
    


    for i in 0..ANALYZER_NUM_BINS {
        let freq = bin_to_freq(i, sample_rate, ANALYZER_FFT_SIZE);
        if freq < FREQ_MIN { continue; }
        if freq > freq_max { break; }
        
        let x = rect.min.x + freq_to_x(freq, FREQ_MIN, freq_max, rect.width());
        let y = db_to_y(mags[i], rect);
        
        points.push(Pos2::new(x, y));
    }

    points
}

/// Create a filled mesh from spectrum path points.
fn create_spectrum_fill(points: &[Pos2], bottom_y: f32, color: Color32) -> Mesh {
    let mut mesh = Mesh::default();
    if points.len() < 2 { return mesh; }
    
    for i in 0..points.len() {
        mesh.vertices.push(Vertex {
            pos: points[i],
            uv: egui::epaint::WHITE_UV,
            color,
        });
        mesh.vertices.push(Vertex {
            pos: Pos2::new(points[i].x, bottom_y),
            uv: egui::epaint::WHITE_UV,
            color,
        });
    }
    
    for i in 0..(points.len() - 1) {
        let base = (i * 2) as u32;
        // Two triangles per segment
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
        mesh.indices.extend_from_slice(&[base + 1, base + 3, base + 2]);
    }
    
    mesh
}
