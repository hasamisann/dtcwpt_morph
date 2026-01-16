use nih_plug::prelude::*;
use nih_plug_egui::egui::{self, Color32, RichText, Rect, Pos2, Stroke, CornerRadius, UiBuilder};
use nih_plug_egui::{create_egui_editor, EguiState};

use std::sync::Arc;

use crate::{DtcwptMorphParams, SharedTopologyState};

const WINDOW_WIDTH: u32 = 280;
const WINDOW_HEIGHT: u32 = 360;

// Colors
const BG_COLOR: Color32 = Color32::from_rgb(20, 20, 28);
const HEADER_COLOR: Color32 = Color32::from_rgb(30, 30, 40);
const ACCENT_COLOR: Color32 = Color32::from_rgb(180, 160, 100); // Gold accent
const TEXT_COLOR: Color32 = Color32::from_rgb(220, 220, 220);
const TEXT_DIM: Color32 = Color32::from_rgb(150, 150, 150);

/// Create the editor state (window size, etc.)
pub fn default_state() -> Arc<EguiState> {
    EguiState::from_size(WINDOW_WIDTH, WINDOW_HEIGHT)
}

// Initial state type
#[derive(Clone, Copy, PartialEq)]
enum View {
    Main,
    Settings,
    About,
}

struct GuiState {
    view: View,
    // We need editor_state to resize window
    // safe to keep reference? yes Arc.
    // actually create_egui_editor takes Arc<EguiState> and passes it to us? 
    // No, the closure only gets `egui_ctx` and `setter`. 
    // We need to move the Arc into the closure state.
}

/// Create the egui editor.
pub fn create(
    params: Arc<DtcwptMorphParams>,
    editor_state: Arc<EguiState>,
    topology_state: Arc<SharedTopologyState>,
    sample_rate: f32, // Added
) -> Option<Box<dyn Editor>> {
    create_egui_editor(
        editor_state.clone(), // Clone for the closure
        GuiState { view: View::Main },
        |_, _| {},
        move |egui_ctx, setter, state| {
            // Fixed Size Window - No Resizing
            egui::CentralPanel::default()
                .frame(egui::Frame::default().fill(BG_COLOR)) // Remove default margin (8px)
                .show(egui_ctx, |ui| {
                     // Main Layout Logic
                     ui.set_min_size(egui::vec2(WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32));
                     
                     match state.view {
                        View::Main => {
                            ui.vertical(|ui| {
                                // ========== HEADER ==========
                                egui::Frame::default()
                                    .fill(HEADER_COLOR)
                                    .inner_margin(egui::Margin::symmetric(12, 8))
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(RichText::new("DT-CWPT Morph").size(14.0).strong().color(Color32::WHITE));
                                            
                                            // Right aligned buttons
                                            ui.allocate_ui_with_layout(
                                                ui.available_size(),
                                                egui::Layout::right_to_left(egui::Align::Center), 
                                                |ui| {
                                                    ui.spacing_mut().item_spacing.x = 12.0; // Match frame margin
                                                    // Info -> About
                                                    if ui.button(RichText::new("ℹ").color(TEXT_COLOR)).clicked() {
                                                        state.view = View::About;
                                                    }
                                                    // Gear -> Settings
                                                    if ui.button(RichText::new("⚙").color(TEXT_COLOR)).clicked() {
                                                        state.view = View::Settings;
                                                    }
                                                }
                                            );
                                        });
                                    });

                                ui.add_space(10.0);

                                // ========== ANALYZER ==========
                                egui::Frame::default()
                                    .fill(Color32::from_rgb(25, 25, 35))
                                    .corner_radius(4.0)
                                    .inner_margin(10)
                                    .show(ui, |ui| {
                                        ui.set_height(140.0); // Fixed Height
                                        ui.set_width(ui.available_width());
                                        ui.vertical_centered(|ui| {
                                             ui.add_space(60.0); 
                                             ui.label(RichText::new("Analyzer Placeholder").color(TEXT_DIM));
                                        });
                                    });

                                ui.add_space(15.0);

                                // ========== PARAMETERS ==========
                                // No wrapper margin, let columns fill width
                                egui::Frame::default().inner_margin(0).show(ui, |ui| {
                                    ui.vertical(|ui| {
                                        // Sliders row - Grid Layout (3 columns)
                                        ui.columns(3, |cols| {
                                            // Magnitude
                                            cols[0].vertical_centered(|ui| {
                                                let mut mag = params.mag.value();
                                                let response = ui.add(crate::knob::Knob::new(&mut mag, 0.0, 1.0, crate::knob::KnobStyle::Wiper)
                                                    .with_size(30.0)
                                                    .with_sweep_range(0.125, 0.75)
                                                    .with_label("Mag", crate::knob::LabelPosition::Bottom)
                                                    .with_colors(TEXT_DIM, ACCENT_COLOR, TEXT_DIM)
                                                    .with_drag_sensitivity(0.015));
                                                if response.changed() {
                                                    setter.begin_set_parameter(&params.mag);
                                                    setter.set_parameter(&params.mag, mag);
                                                    setter.end_set_parameter(&params.mag);
                                                }
                                                response.context_menu(|ui| {
                                                    if ui.add(egui::DragValue::new(&mut mag).speed(0.01).range(0.0..=1.0)).changed() {
                                                        setter.begin_set_parameter(&params.mag);
                                                        setter.set_parameter(&params.mag, mag);
                                                        setter.end_set_parameter(&params.mag);
                                                    }
                                                });
                                            });
    
                                            // Phase
                                            cols[1].vertical_centered(|ui| {
                                                let mut phase = params.phase.value();
                                                let response = ui.add(crate::knob::Knob::new(&mut phase, 0.0, 1.0, crate::knob::KnobStyle::Wiper)
                                                    .with_size(30.0)
                                                    .with_sweep_range(0.125, 0.75)
                                                    .with_label("Phase", crate::knob::LabelPosition::Bottom)
                                                    .with_colors(TEXT_DIM, ACCENT_COLOR, TEXT_DIM)
                                                    .with_drag_sensitivity(0.015));
                                                if response.changed() {
                                                    setter.begin_set_parameter(&params.phase);
                                                    setter.set_parameter(&params.phase, phase);
                                                    setter.end_set_parameter(&params.phase);
                                                }
                                                response.context_menu(|ui| {
                                                    if ui.add(egui::DragValue::new(&mut phase).speed(0.01).range(0.0..=1.0)).changed() {
                                                        setter.begin_set_parameter(&params.phase);
                                                        setter.set_parameter(&params.phase, phase);
                                                        setter.end_set_parameter(&params.phase);
                                                    }
                                                });
                                            });
    
                                            // Threshold
                                            cols[2].vertical_centered(|ui| {
                                                let mut thr = params.threshold.value();
                                                let response = ui.add(crate::knob::Knob::new(&mut thr, -60.0, 0.0, crate::knob::KnobStyle::Wiper)
                                                    .with_size(30.0)
                                                    .with_sweep_range(0.125, 0.75)
                                                    .with_label("Thres", crate::knob::LabelPosition::Bottom)
                                                    .with_colors(TEXT_DIM, ACCENT_COLOR, TEXT_DIM)
                                                    .with_drag_sensitivity(0.015)
                                                    .with_label_format(|v| format!("{:.1} dB", v)));
                                                if response.changed() {
                                                    setter.begin_set_parameter(&params.threshold);
                                                    setter.set_parameter(&params.threshold, thr);
                                                    setter.end_set_parameter(&params.threshold);
                                                }
                                                response.context_menu(|ui| {
                                                    if ui.add(egui::DragValue::new(&mut thr).speed(1.0).range(-60.0..=0.0).suffix(" dB")).changed() {
                                                        setter.begin_set_parameter(&params.threshold);
                                                        setter.set_parameter(&params.threshold, thr);
                                                        setter.end_set_parameter(&params.threshold);
                                                    }
                                                });
                                            });
                                        });
    
                                        ui.add_space(15.0);
    
                                        // Bypass row
                                        ui.horizontal(|ui| {
                                            ui.add_space(20.0);
                                            let mut bypass_low = params.bypass_low.value();
                                            let mut bypass_high = params.bypass_high.value();
    
                                            ui.label(RichText::new("Bypass:").size(12.0).color(TEXT_COLOR));
                                            ui.add_space(10.0);
    
                                            if ui.checkbox(&mut bypass_low, "Low").changed() {
                                                setter.begin_set_parameter(&params.bypass_low);
                                                setter.set_parameter(&params.bypass_low, bypass_low);
                                                setter.end_set_parameter(&params.bypass_low);
                                            }
                                            ui.add_space(20.0);
                                            if ui.checkbox(&mut bypass_high, "High").changed() {
                                                setter.begin_set_parameter(&params.bypass_high);
                                                setter.set_parameter(&params.bypass_high, bypass_high);
                                                setter.end_set_parameter(&params.bypass_high);
                                            }
                                        });
                                    });
                                });
                            });
                        }
                        
                        View::Settings => {
                            ui.vertical(|ui| {
                                // Header
                                egui::Frame::default()
                                    .fill(HEADER_COLOR)
                                    .inner_margin(egui::Margin::symmetric(12, 8))
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(RichText::new("Splitter Editor").size(14.0).strong().color(Color32::WHITE));
                                            
                                            ui.allocate_ui_with_layout(
                                                ui.available_size(),
                                                egui::Layout::right_to_left(egui::Align::Center), 
                                                |ui| {
                                                     // Widen Back button (approx 1.2x)
                                                     ui.spacing_mut().button_padding.x *= 1.5; 
                                                     if ui.button(RichText::new("Back").color(TEXT_COLOR)).clicked() {
                                                         state.view = View::Main;
                                                     }
                                                }
                                            );
                                        });
                                    });

                                // Editor Content - fills remaining space
                                let available_size = ui.available_size(); // Available in fixed window
                                let (rect, _) = ui.allocate_exact_size(available_size, egui::Sense::hover());
                                draw_topology_editor(ui, rect, &topology_state, sample_rate);
                            });
                        }
                        
                        View::About => {
                            ui.vertical(|ui| {
                                // Header
                                egui::Frame::default()
                                    .fill(HEADER_COLOR)
                                    .inner_margin(egui::Margin::symmetric(12, 8))
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(RichText::new("About").size(14.0).strong().color(Color32::WHITE));
                                            
                                            ui.allocate_ui_with_layout(
                                                ui.available_size(),
                                                egui::Layout::right_to_left(egui::Align::Center), 
                                                |ui| {
                                                     // Widen Back button (approx 1.2x)
                                                     ui.spacing_mut().button_padding.x *= 1.5;
                                                     if ui.button(RichText::new("Back").color(TEXT_COLOR)).clicked() {
                                                         state.view = View::Main;
                                                     }
                                                }
                                            );
                                        });
                                    });
                                
                                // About Content
                                ui.vertical_centered(|ui| {
                                    ui.add_space(60.0);
                                    ui.label(RichText::new("DT-CWPT Morph").size(20.0).strong().color(Color32::WHITE)); 
                                    ui.label(RichText::new("Beta Release").size(10.0).color(TEXT_DIM));
                                    ui.add_space(30.0);
                                    
                                    if ui.hyperlink_to(RichText::new("GitHub Repository").size(12.0), "https://github.com/hasamisann/dtcwpt_morph").clicked() { /* */ }
                                    ui.add_space(10.0);
                                    if ui.hyperlink_to(RichText::new("X (Twitter): @LTSU_n_nv").size(12.0), "https://x.com/LTSU_n_nv").clicked() { /* */ }
                                    ui.add_space(10.0);
                                    if ui.hyperlink_to(RichText::new("SoundCloud: @LTSU_n_nv").size(12.0), "https://soundcloud.com/LTSU_n_nv").clicked() { /* */ }
                                    
                                    ui.add_space(50.0);
                                    ui.label(RichText::new("VST is a registered trademark of\nSteinberg Media Technologies GmbH").size(9.0).color(TEXT_DIM));
                                });
                            });
                        }
                    }
                });
        },
    )
}

fn draw_topology_editor(ui: &mut egui::Ui, rect: Rect, state: &SharedTopologyState, sample_rate: f32) {
    let max_depth = 8;
    
    // Horizontal Layout:
    // X axis = Depth (Root Left -> Children Right)
    // Y axis = Frequency (Low Bottom -> High Top, usually)
    // "Frequency axis is on the right" -> We place axis at rect.max.x

    let id = ui.id().with("topo_editor_v2");
    // State: offset (pixels), scale (multiplier)
    #[derive(Clone, Copy)]
    struct ViewState { offset: egui::Vec2, scale: f32 }
    let mut view = ui.data_mut(|d| d.get_temp::<ViewState>(id))
        .unwrap_or(ViewState { 
            offset: egui::vec2(0.0, -30.0), 
            scale: 0.9
        });

    // Interact for Pan/Zoom
    let response = ui.interact(rect, id, egui::Sense::drag());
    
    // Zoom
    if response.hovered() {
        let zoom_delta = ui.input(|i| i.zoom_delta());
        if zoom_delta != 1.0 {
            let pointer = ui.input(|i| i.pointer.hover_pos()).unwrap_or(rect.center()) - rect.min.to_vec2();
            
            // Calculate new scale with clamping
            let new_scale = (view.scale * zoom_delta).clamp(0.1, 10.0);
            let effective_zoom = new_scale / view.scale;
            
            // X Axis: Origin is Left (rect.min.x). Pointer is already relative to Left.
            view.offset.x = pointer.x + (view.offset.x - pointer.x) * effective_zoom;

            // Y Axis: Origin is Bottom (rect.max.y).
            // pointer.y is relative to Top. Convert to relative to Bottom.
            let pivot_y = pointer.y - rect.height();
            view.offset.y = pivot_y + (view.offset.y - pivot_y) * effective_zoom;

            view.scale = new_scale;
        }
    }
    
    // Pan
    if response.dragged() {
        view.offset += response.drag_delta();
    }
    
    ui.data_mut(|d| d.insert_temp(id, view));

    // Clip
    let painter = ui.painter().with_clip_rect(rect);
    
    // Instructions Overlay
    painter.text(
        rect.min + egui::vec2(10.0, 10.0),
        egui::Align2::LEFT_TOP,
        "Left Click: Split/Merge\nDrag: Pan\nScroll: Zoom",
        egui::FontId::proportional(12.0),
        TEXT_DIM.gamma_multiply(0.8),
    );

    // Get current destinations
    let mut current_dests = {
        let config = state.config.lock().unwrap();
        config.destinations.clone()
    };
    let mut changed = false;

    // Helper: Depth to X (0..1)
    let get_x_for_depth = |d: usize| -> f32 {
        // Linear or Geometric? 
        // 280px width is tight. Simple linear might be best or slight compression.
        // Let's use geometric to allow more space for root? Or vice versa?
        // Usually deep levels get crowded Y-wise. X-wise we have consistent branching.
        // Linear X is probably cleanest.
        
        let margin_left = 0.05;
        let margin_right = 0.20; // Room for axis Labels
        let useful = 1.0 - margin_left - margin_right;
        
        margin_left + (d as f32 / max_depth as f32) * useful
    };

    // Helper: Logical (Freq 0..1, Depth 0..1) -> Screen Pos
    let to_screen = |freq: f32, depth_norm: f32| -> Pos2 {
        // X = Depth (Left to Right)
        let base_x = depth_norm * rect.width();
        let screen_x = rect.min.x + view.offset.x + base_x * view.scale;
        
        // Y = Frequency (Bottom to Top for Low->High)
        // 0.0 freq = Bottom, 1.0 freq = Top
        // In screen Y: Bottom is max_y, Top is min_y.
        // base_y_from_bottom = freq * rect.height();
        // screen_y = rect.max.y + view.offset.y - base_y_from_bottom * view.scale;
        
        let base_y_from_bottom = freq * rect.height();
        let screen_y = rect.max.y + view.offset.y - base_y_from_bottom * view.scale;
        
        Pos2::new(screen_x, screen_y)
    };

    // Recursive Drawing
    fn draw_node(
        painter: &egui::Painter,
        ui: &egui::Ui,
        to_screen: &dyn Fn(f32, f32) -> Pos2,
        path: String,
        depth: usize,
        max_depth: usize,
        index: usize,
        freq_center: f32, // 0..1
        freq_width: f32, // 0..1
        current_dests: &mut Vec<String>,
        changed: &mut bool,
        sample_rate: f32,
        get_x_for_depth: &dyn Fn(usize) -> f32,
    ) {
        let mut is_leaf = false;
        let mut is_splitter = false;
        
        if current_dests.contains(&path) {
            is_leaf = true;
        } else {
            is_splitter = true; 
        }

        let depth_x = get_x_for_depth(depth);
        let screen_pos = to_screen(freq_center, depth_x);
        
        // Draw connections to children
        if is_splitter && depth < max_depth {
            let next_depth_x = get_x_for_depth(depth + 1);
            
            // Left Child (Low Freq) -> Lower in value (0..1), so closer to 0. 
            // In our Y mapping (Low=Bottom), 0 is Bottom. 
            // So "Low Freq" child is visually Below (Higher Y pixel value).
            // Wait, freq 0 is Bottom. freq 0.25 is "Higher" than freq 0.
            // freq 0.75 is "Higher" than freq 0.25.
            
            // L Child = Center - Width/4. (Lower frequency).
            // R Child = Center + Width/4. (Higher frequency).
            
            // Left Child (Low Freq)
            let l_idx = index << 1;
            let l_freq = freq_center - freq_width / 4.0;
            let l_screen = to_screen(l_freq, next_depth_x);
            
            painter.line_segment([screen_pos, l_screen], Stroke::new(1.0, TEXT_DIM.gamma_multiply(0.3)));
            draw_node(painter, ui, to_screen, path.clone() + "L", depth + 1, max_depth, l_idx, l_freq, freq_width / 2.0, current_dests, changed, sample_rate, get_x_for_depth);

            // Right Child (High Freq)
            let r_idx = (index << 1) | 1;
            let r_freq = freq_center + freq_width / 4.0;
            let r_screen = to_screen(r_freq, next_depth_x);
            
            painter.line_segment([screen_pos, r_screen], Stroke::new(1.0, TEXT_DIM.gamma_multiply(0.3)));
            draw_node(painter, ui, to_screen, path.clone() + "H", depth + 1, max_depth, r_idx, r_freq, freq_width / 2.0, current_dests, changed, sample_rate, get_x_for_depth);
        }

        // Draw Node
        let radius = 6.0; 
        let color = if is_leaf { ACCENT_COLOR } else { BG_COLOR };
        let stroke_color = if is_leaf { Color32::WHITE } else { TEXT_DIM };
        
        let node_rect = Rect::from_center_size(screen_pos, egui::vec2(radius*2.0, radius*2.0));
        let response = ui.interact(node_rect, ui.id().with(index), egui::Sense::click());

        if response.clicked() {
            if is_leaf {
                if depth < max_depth {
                    current_dests.retain(|p| p != &path);
                    current_dests.push(path.clone() + "L");
                    current_dests.push(path.clone() + "H");
                    *changed = true;
                }
            } else if is_splitter {
                let p_len = path.len();
                let has_children = current_dests.iter().any(|p| p.starts_with(&path) && p.len() > p_len);
                if has_children {
                    current_dests.retain(|p| !p.starts_with(&path));
                    current_dests.push(path.clone());
                    *changed = true;
                }
            }
        }

        if response.hovered() {
            painter.circle_filled(screen_pos, radius * 1.5, ACCENT_COLOR.gamma_multiply(0.3));
            response.on_hover_ui(|ui| {
                let nyquist = sample_rate / 2.0;
                let center_hz = freq_center * nyquist;
                let bw_hz = freq_width * nyquist;
                let min_hz = center_hz - bw_hz / 2.0;
                let max_hz = center_hz + bw_hz / 2.0;
                
                ui.label(format!(
                    "Path: {}\nType: {}\nRange: {:.0} Hz - {:.0} Hz", 
                    path, 
                    if is_leaf { "Leaf" } else { "Splitter" },
                    min_hz, max_hz
                ));
            });
        }

        painter.circle_filled(screen_pos, radius, color);
        let final_stroke_color = if is_splitter { stroke_color.gamma_multiply(0.5) } else { stroke_color };
        painter.circle_stroke(screen_pos, radius, Stroke::new(1.0, final_stroke_color));
    }

    // Start drawing (Root at Freq 0.5, Depth 0)
    draw_node(
        &painter, ui, &to_screen, 
        "".to_string(), 0, max_depth, 1, 
        0.5, 1.0, 
        &mut current_dests, &mut changed, sample_rate, &get_x_for_depth
    );
    
    // Axis (Right Side)
    // Vertical line at rect.max.x - margin
    let axis_x = rect.max.x - 30.0; // Fixed visual position?
    // Wait, if we use zoom/pan, logical axis position?
    // "Frequency axis is on the right".
    // Does it move with nodes? Or is it fixed overlay?
    // Usually fixed overlay. But if nodes pan/zoom, axis labels must match.
    // The previous implementation inferred ticks from logical coords.
    
    // Draw Axis line (Fixed UI element or Logical?)
    // Let's make it fixed in Screen X, but Ticks move in Screen Y.
    
    let axis_top = rect.min.y;
    let axis_bottom = rect.max.y;
    
    painter.line_segment(
        [Pos2::new(axis_x, axis_bottom), Pos2::new(axis_x, axis_top)], 
        Stroke::new(1.0, TEXT_DIM)
    );
    
    // Adaptive Ticks Y
    let nyquist = sample_rate / 2.0;
    
    // Height in pixels mapping 0..Nyquist
    // 0Hz = rect.max.y + view.offset.y
    // Nyq = rect.max.y + view.offset.y - rect.height() * view.scale
    
    // Screen height range for signals: rect.height() * view.scale.
    // Pixel range = rect.height() * view.scale.
    // HZ range = nyquist.
    // Px per Hz = (rect.height() * view.scale) / nyquist.
    
    let px_per_hz = (rect.height() * view.scale) / nyquist;
    let min_px_step = 40.0;
    
    if px_per_hz > 0.0001 { // Prevent div/0
        let min_hz_step = min_px_step / px_per_hz;
        
        let magnitude = 10.0f32.powf(min_hz_step.log10().floor());
        let residual = min_hz_step / magnitude;
        let nice_step = if residual > 5.0 { 10.0 * magnitude } 
                        else if residual > 2.0 { 5.0 * magnitude } 
                        else { 2.0 * magnitude };
        
        let start_hz = 0.0;
        let mut curr_hz = start_hz;
        
        while curr_hz <= nyquist {
            let t = curr_hz / nyquist; // 0..1
            
            // Map t (freq) to Screen Y
            let y = rect.max.y + view.offset.y - (t * rect.height() * view.scale);
            
            if rect.y_range().contains(y) {
                let tick_pos = Pos2::new(axis_x, y);
                painter.line_segment([tick_pos, tick_pos + egui::vec2(5.0, 0.0)], Stroke::new(1.0, TEXT_DIM));
                
                let text = if curr_hz >= 1000.0 {
                    format!("{:.1}k", curr_hz / 1000.0)
                } else {
                    format!("{:.0}", curr_hz)
                };
                
                painter.text(
                    tick_pos + egui::vec2(8.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    text,
                    egui::FontId::proportional(10.0),
                    TEXT_DIM,
                );
            }
            curr_hz += nice_step;
        }
    }

    if changed {
        let mut config = state.config.lock().unwrap();
        config.destinations = current_dests;
        state.dirty.store(true, std::sync::atomic::Ordering::Release);
    }
}
