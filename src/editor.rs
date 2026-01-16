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
                                                    // Gear -> Settings
                                                    if ui.button(RichText::new("⚙").color(TEXT_COLOR)).clicked() {
                                                        state.view = View::Settings;
                                                    }
                                                    // Info -> About
                                                    if ui.button(RichText::new("ℹ").color(TEXT_COLOR)).clicked() {
                                                        state.view = View::About;
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
    
    // Frequency Axis
    // We want to show a guide logic.
    // X = 0.0 -> 0 Hz
    // X = 1.0 -> sample_rate / 2.0 Hz
    // We use a persistent ID to store transform state if possible, or just default.
    // Egui's `scroll_to_me` or `Window` are options, but implementing simple drag/zoom is manual.
    // For simplicity in this iteration: Fixed view with "Fit to Screen" layout logic that handles depth 8.
    // Depth 8 has 256 leaves. That requires ~4000px width for distinct nodes if strictly linear.
    // We will use a virtual coordinate system and transform user inputs.

    let id = ui.id().with("topo_editor");
    let mut transform = ui.data_mut(|d| d.get_temp::<egui::emath::RectTransform>(id))
        .unwrap_or_else(|| egui::emath::RectTransform::identity(rect));

    // Interact for Pan/Zoom
    let response = ui.interact(rect, id, egui::Sense::drag());
    
    // Handle Zoom (Scroll)
    if response.hovered() {
        let zoom_delta = ui.input(|i| i.zoom_delta());
        if zoom_delta != 1.0 {
            let _pointer = ui.input(|i| i.pointer.hover_pos()).unwrap_or(rect.center());
            // Scale around pointer
            let scale = zoom_delta;
            let new_scale = transform.scale() * scale;
            // Limit zoom
            if new_scale.x > 0.1 && new_scale.x < 10.0 {
                let _new_transform = transform.clone();
                // Simple scaling approach: just scale the `to` rect?
                // RectTransform maps `from` (normalized 0..1 or logical) to `to` (screen).
                // Let's implement manual offset/scale.
            }
        }
    }
    
    // Manual transform management
    // state: (offset: Vec2, scale: f32)
    #[derive(Clone, Copy)]
    struct ViewState { offset: egui::Vec2, scale: f32 }
    let mut view = ui.data_mut(|d| d.get_temp::<ViewState>(id))
        .unwrap_or(ViewState { 
            offset: egui::vec2(rect.width() * 0.025, rect.height() * 0.025), 
            scale: 0.95 
        });

    if response.dragged() {
        view.offset += response.drag_delta();
    }
    if response.hovered() {
        let zoom = ui.input(|i| i.zoom_delta());
        if zoom != 1.0 {
            let pointer = ui.input(|i| i.pointer.hover_pos()).unwrap_or(rect.center()) - rect.min.to_vec2();
            // Zoom towards pointer:
            // new_offset = offset + (pointer - offset) * (1 - zoom)
            // But dragging moves offset. Scale affects drawing.
            // Let's optimize: simple centered zoom for now or just scale.
            
            // Zoom centered relative to pointer in screen space?
            // Easier: Adjust scale, adjust offset to keep pointer stable.
            // p_screen = Rect.min + offset + (logical * rect_size * scale)
            // relative_p = p_screen - Rect.min = offset + (logical * rect_size * scale)
            // We want to keep p_screen constant under mouse.
            
            // Let's simplify: view.offset and view.scale apply to the "Rect-normalized" space.
            // visual_pos = func(logical_pos)
            // visual_pos = offset + logical_pos * size * scale
            // When scaling:
            // p_rel = visual_pos under mouse relative to rect.min
            // p_rel = offset + logical_under_mouse * size * scale
            // If scale changes to scale_new, and we want p_rel to stay same:
            // offset_new + logical_under_mouse * size * scale_new = offset + logical_under_mouse * size * scale
            // logical_under_mouse * size = (p_rel - offset) / scale
            // offset_new = p_rel - (p_rel - offset) / scale * scale_new
            // offset_new = p_rel - (p_rel - offset) * (scale_new / scale)
            // offset_new = p_rel * (1 - zoom_factor) + offset * zoom_factor   <-- This was correct IF p_rel is in same coord system as offset
            
            // Yes, p_rel is pixel offset from rect.min. view.offset is pixel offset.
            // So the math holds.
            
            let p_rel = pointer;
            view.offset = p_rel.to_vec2() * (1.0 - zoom) + view.offset * zoom;
            view.scale *= zoom;
        }
    }
    ui.data_mut(|d| d.insert_temp(id, view));

    // Clip to rect
    let painter = ui.painter().with_clip_rect(rect);

    // Get current destinations
    let mut current_dests = {
        let config = state.config.lock().unwrap();
        config.destinations.clone()
    };

    let mut changed = false;

    // Helper: Screen projection
    let to_screen = |logical_pos: Pos2| -> Pos2 {
        let base_x = logical_pos.x * rect.width();
        
        // Variable vertical compression:
        // Use a power law or exponential decay for Y steps?
        // Let y_linear = logical_pos.y
        // We want step size to decrease.
        // But draw_node calculates logical_pos.y assuming linear steps. 
        // We should adjust logical_pos.y calculation in draw_node instead!
        // So here we keep linear mapping.
        
        let base_y = logical_pos.y * rect.height();
        rect.min + view.offset + egui::vec2(base_x, base_y) * view.scale
    };

    // Helper: Y position for depth
    let get_y_for_depth = |d: usize| -> f32 {
        // Linear: d / (max + 1)
        // Compressed: sum(0.9^i)
        // Let's use simple geometric series sum logic. 
        // step[i] = base * factor^i
        // y[d] = sum(step[0]..step[d-1])
        // total_height = sum(step[0]..step[max])
        
        let base = 1.0;
        let factor: f32 = 0.85; // Compression factor
        
        let mut y = 0.0;
        for i in 0..d {
            y += base * factor.powf(i as f32);
        }
        
        // Normalize
        let mut total = 0.0;
        for i in 0..=max_depth {
            total += base * factor.powf(i as f32);
        }
        
        // Add margins
        let margin_top = 0.05;
        let margin_bot = 0.05;
        let useful = 1.0 - margin_top - margin_bot;
        
        margin_top + (y / total) * useful
    };

    // Recursive Drawing
    // We draw from Root. Logical coords: X [0..1], Y [0..1]
    fn draw_node(
        painter: &egui::Painter,
        ui: &egui::Ui,
        to_screen: &dyn Fn(Pos2) -> Pos2,
        path: String,
        depth: usize,
        max_depth: usize,
        index: usize,
        logical_pos: Pos2,
        logical_width: f32, // Width covered by this node's subtree (at this depth)
        current_dests: &mut Vec<String>,
        changed: &mut bool,
        sample_rate: f32, // Added
        get_y_for_depth: &dyn Fn(usize) -> f32, // Passed closure
    ) {
        let _is_dest = current_dests.contains(&path);
        
        // Check if children exist
        // Efficient check: Does any dest start with `path`?
        let mut is_leaf = false;
        let mut is_splitter = false;
        
        if current_dests.contains(&path) {
            is_leaf = true;
        } else {
            is_splitter = true; 
        }

        let screen_pos = to_screen(logical_pos);
        
        // Draw connections to children if split
        if is_splitter && depth < max_depth {
            let _y_curr = get_y_for_depth(depth);
            let y_next = get_y_for_depth(depth + 1);
            
            // Left Child
            let l_idx = index << 1;
            let l_log_pos = Pos2::new(logical_pos.x - logical_width / 4.0, y_next);
            let l_screen = to_screen(l_log_pos);
            
            painter.line_segment([screen_pos, l_screen], Stroke::new(1.0, TEXT_DIM.gamma_multiply(0.3)));
            draw_node(painter, ui, to_screen, path.clone() + "L", depth + 1, max_depth, l_idx, l_log_pos, logical_width / 2.0, current_dests, changed, sample_rate, get_y_for_depth);

            // Right Child
            let r_idx = (index << 1) | 1;
            let r_log_pos = Pos2::new(logical_pos.x + logical_width / 4.0, y_next);
            let r_screen = to_screen(r_log_pos);
            
            painter.line_segment([screen_pos, r_screen], Stroke::new(1.0, TEXT_DIM.gamma_multiply(0.3)));
            draw_node(painter, ui, to_screen, path.clone() + "H", depth + 1, max_depth, r_idx, r_log_pos, logical_width / 2.0, current_dests, changed, sample_rate, get_y_for_depth);
        }

        // Draw Node
        let radius = 6.0; // Scalable?
        let color = if is_leaf { ACCENT_COLOR } else { BG_COLOR };
        let stroke_color = if is_leaf { Color32::WHITE } else { TEXT_DIM };
        
        // Interaction
        let node_rect = Rect::from_center_size(screen_pos, egui::vec2(radius*2.0, radius*2.0));
        let response = ui.interact(node_rect, ui.id().with(index), egui::Sense::click());

        if response.clicked() {
            if is_leaf {
                // SPLIT ACTION
                if depth < max_depth {
                    // Remove self, add children
                    current_dests.retain(|p| p != &path);
                    current_dests.push(path.clone() + "L");
                    current_dests.push(path.clone() + "H");
                    *changed = true;
                }
            } else if is_splitter {
                // MERGE ACTION
                // Remove all descendants, add self
                let p_len = path.len();
                // Check if any children exist before merging
                let has_children = current_dests.iter().any(|p| p.starts_with(&path) && p.len() > p_len);
                if has_children {
                    current_dests.retain(|p| !p.starts_with(&path)); // Remove children
                    current_dests.push(path.clone()); // Add self
                    *changed = true;
                }
            }
        }

        if response.hovered() {
            painter.circle_filled(screen_pos, radius * 1.5, ACCENT_COLOR.gamma_multiply(0.3));
            response.on_hover_ui(|ui| {
                // Calculate frequency range
                // x center is logical_pos.x [0..1]
                // logical_width covers the band.
                let nyquist = sample_rate / 2.0;
                let center_hz = logical_pos.x * nyquist;
                let bw_hz = logical_width * nyquist;
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
        // Dim splitters
        let final_stroke_color = if is_splitter { stroke_color.gamma_multiply(0.5) } else { stroke_color };
        painter.circle_stroke(screen_pos, radius, Stroke::new(1.0, final_stroke_color));
    }

    // Start drawing
    draw_node(
        &painter, 
        ui, 
        &to_screen, 
        "".to_string(), 
        0, 
        max_depth, 
        1, 
        Pos2::new(0.5, get_y_for_depth(0)), 
        1.0, 
        &mut current_dests, 
        &mut changed,
        sample_rate,
        &get_y_for_depth,
    );
    
    // Draw Axis (Fixed at bottom of view)
    let axis_y = rect.max.y - 20.0;
    
    // Project logical X (0..1) to screen X
    let to_screen_x = |lx: f32| -> f32 {
        rect.min.x + view.offset.x + lx * rect.width() * view.scale
    };

    let start_x = to_screen_x(0.0);
    let end_x = to_screen_x(1.0);
    
    // Axis visible range
    // We only draw if the axis line intersects the view rect horizontally
    // But conceptually the axis is infinite or 0..1? 0..1.
    
    // Draw background for axis?
    // painter.rect_filled(Rect::from_x_y_ranges(rect.x_range(), (axis_y-10.0)..=rect.max.y), 0.0, Color32::from_black_alpha(100));

    painter.line_segment([Pos2::new(start_x, axis_y), Pos2::new(end_x, axis_y)], Stroke::new(1.0, TEXT_DIM));
    
    // Adaptive Ticks
    // Nyquist Hz = sample_rate / 2.0
    // We want ticks at nice Hz intervals.
    let nyquist = sample_rate / 2.0;
    let width_px = end_x - start_x;
    
    // How many pixels per 1kHz?
    // px_per_hz = width_px / nyquist;
    // We want min 50px between ticks.
    // min_hz_step = 50.0 / px_per_hz = 50.0 * nyquist / width_px
    
    let min_px_step = 60.0;
    if width_px > 1.0 {
        let min_hz_step = min_px_step * nyquist / width_px;
        
        // Find nice step (100, 500, 1000, 5000...)
        let magnitude = 10.0f32.powf(min_hz_step.log10().floor());
        let residual = min_hz_step / magnitude;
        let nice_step = if residual > 5.0 {
            10.0 * magnitude
        } else if residual > 2.0 {
            5.0 * magnitude
        } else {
            2.0 * magnitude // or 1.0
        };
        
        let start_hz = 0.0;
        let mut curr_hz = start_hz;
        
        while curr_hz <= nyquist {
            let t = curr_hz / nyquist;
            let x = to_screen_x(t);
            
            if rect.x_range().contains(x) {
                let tick_pos = Pos2::new(x, axis_y);
                painter.line_segment([tick_pos, tick_pos + egui::vec2(0.0, 5.0)], Stroke::new(1.0, TEXT_DIM));
                
                let text = if curr_hz >= 1000.0 {
                    format!("{:.1}k", curr_hz / 1000.0)
                } else {
                    format!("{:.0}", curr_hz)
                };
                
                painter.text(
                    tick_pos + egui::vec2(0.0, 8.0),
                    egui::Align2::CENTER_TOP,
                    text,
                    egui::FontId::proportional(10.0),
                    TEXT_DIM,
                );
            }
            curr_hz += nice_step;
        }
    }
     // Removed Text Controls Instructions for cleaner view
     // painter.text(...)

    if changed {
        let mut config = state.config.lock().unwrap();
        config.destinations = current_dests;
        state.dirty.store(true, std::sync::atomic::Ordering::Release);
    }
}
