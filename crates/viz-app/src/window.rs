//! Window + GPU context and the winit event loop.
//!
//! Implements the winit 0.30 [`ApplicationHandler`] pattern: the [`Window`] and the
//! whole wgpu chain (Instance → Surface → Adapter → Device/Queue) are created in
//! [`ApplicationHandler::resumed`] on the main thread (winit requires the window to
//! be created from the event loop, not before it starts). The surface is
//! configured for low-latency vsync presentation: [`wgpu::PresentMode::Fifo`] with
//! `desired_maximum_frame_latency = 1`, the surface-preferred format, sized in
//! physical pixels.
//!
//! Frame pacing: macOS throttles `request_redraw` to the display link, but
//! on >60 Hz displays (ProMotion) that can be 120 Hz. We enforce a HARD 60 Hz update
//! rate by skipping presents that arrive sooner than [`MIN_FRAME_INTERVAL`] since the
//! last present, while still re-requesting the next redraw so the loop keeps running.
//! The skip decision is the pure function [`should_present`] so it can be unit-tested.
//!
//! The visualizer pass is injected through the [`RenderCallback`] seam: this shell
//! renders a dark clear pass into the surface texture FIRST, then hands the same
//! command encoder + the surface view to the callback (the egui overlay today, the
//! visualizer + overlay later) which composites on top with `LoadOp::Load`
//! (loading rather than clearing, so the overlay never wipes the scene beneath it).

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::window::{Window, WindowId, WindowLevel};

/// Hard 60 Hz frame budget. Presents closer together than this are skipped so the
/// effective update rate stays at 60 even on 120 Hz displays.
///
/// Slightly under 1/60 s (≈16.67 ms) to avoid rounding a legitimate 60 Hz vsync into
/// a skip — we use 15.5 ms.
pub const MIN_FRAME_INTERVAL: Duration = Duration::from_micros(15_500);

/// Initial logical window size (physical size is derived via the scale factor).
const INITIAL_WIDTH: u32 = 1280;
const INITIAL_HEIGHT: u32 = 720;

/// Base `1x` logical window size (16:9 compact floating widget).
const PRESET_1X_WIDTH: u32 = 320;
const PRESET_1X_HEIGHT: u32 = 180;

/// Logical pixel tolerance when matching the current window size against a fixed preset.
const SIZE_PRESET_TOLERANCE: f64 = 4.0;

/// Logical pixel tolerance when matching the current window size against the monitor size
/// (covers borderless full-display maximize).
const MAX_MONITOR_TOLERANCE: f64 = 8.0;

/// Live GPU resources shared with the render path.
///
/// Held behind the app so the render callback can borrow `device`/`queue`/`config`
/// without re-acquiring them each frame. No per-frame allocation lives here
/// (the swapchain texture is acquired transiently in [`App::render`]).
pub struct GpuContext {
    /// The presentation surface bound to the window. `'static` because the surface
    /// owns an `Arc<Window>` (created via [`wgpu::SurfaceTarget`] from the Arc).
    pub surface: wgpu::Surface<'static>,
    /// Logical GPU device.
    pub device: wgpu::Device,
    /// Command submission queue.
    pub queue: wgpu::Queue,
    /// Current surface configuration (format, size, present mode). Reconfigured on
    /// resize / scale-factor change.
    pub config: wgpu::SurfaceConfiguration,
    /// The chosen surface texture format (cached for pipeline/overlay construction).
    pub format: wgpu::TextureFormat,
}

impl GpuContext {
    /// Builds the full wgpu chain for `window` at the given physical size.
    ///
    /// Blocks on adapter/device acquisition via `pollster` (one-time setup cost,
    /// off the hot path).
    fn new(window: Arc<Window>, size: PhysicalSize<u32>) -> Self {
        // `InstanceDescriptor` has no `Default` (its `display` field is a boxed
        // trait object); build it via the no-display constructor and force Metal
        // (with a portability fallback) for macOS.
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::METAL | wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(instance_desc);

        let surface = instance
            .create_surface(window)
            .expect("failed to create wgpu surface from window");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .expect("no suitable GPU adapter found");

        // Baseline limits, but raise the texture/resolution caps to what the adapter
        // actually supports: `downlevel_defaults` caps 2D textures at 2048, which is
        // smaller than a Retina window's physical size (e.g. 2560×1440) and would
        // make `Surface::configure` panic on large displays.
        let required_limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("viewmusic-device"),
            required_features: viz_render::device_features(&adapter),
            required_limits,
            ..Default::default()
        }))
        .expect("failed to acquire wgpu device");

        // Start from the surface-preferred config, then override pacing/latency knobs.
        // Clamp to the device's maximum texture extent (same rationale as `resize`).
        let max_dim = device.limits().max_texture_dimension_2d;
        let width = size.width.clamp(1, max_dim);
        let height = size.height.clamp(1, max_dim);
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .expect("surface is incompatible with the selected adapter");
        config.present_mode = wgpu::PresentMode::Fifo;
        config.desired_maximum_frame_latency = 1;
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        // Prefer premultiplied alpha so transparent window mode composites cleanly
        // (Metal supports PreMultiplied and viz-render shaders emit premultiplied RGBA).
        let caps = surface.get_capabilities(&adapter);
        if caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
        {
            config.alpha_mode = wgpu::CompositeAlphaMode::PreMultiplied;
        } else if caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PostMultiplied)
        {
            config.alpha_mode = wgpu::CompositeAlphaMode::PostMultiplied;
        }
        let format = config.format;

        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
            format,
        }
    }

    /// Reconfigures the surface to a new physical size (resize / scale change).
    ///
    /// Zero-sized requests are ignored (minimized window) to avoid wgpu panics,
    /// unchanged clamped dimensions are skipped to avoid redundant swapchain rebuilds,
    /// and dimensions are clamped to the device's maximum texture extent so a move
    /// onto a display larger than the GPU supports degrades instead of panicking.
    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        let max_dim = self.device.limits().max_texture_dimension_2d;
        let new_w = size.width.min(max_dim);
        let new_h = size.height.min(max_dim);
        if self.config.width == new_w && self.config.height == new_h {
            return;
        }
        self.config.width = new_w;
        self.config.height = new_h;
        self.surface.configure(&self.device, &self.config);
    }

    /// Unconditionally reconfigures the surface after `Outdated` or `Lost`.
    fn reconfigure(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        let max_dim = self.device.limits().max_texture_dimension_2d;
        self.config.width = size.width.min(max_dim);
        self.config.height = size.height.min(max_dim);
        self.surface.configure(&self.device, &self.config);
    }
}

/// Context handed to the render callback for one frame after the clear pass.
///
/// Borrows live GPU state plus the acquired surface view. The callback composites
/// on top of the already-cleared scene using `LoadOp::Load`.
pub struct FrameContext<'a> {
    /// GPU device (for transient resources the overlay may need).
    pub device: &'a wgpu::Device,
    /// Submission queue.
    pub queue: &'a wgpu::Queue,
    /// The command encoder already used for the clear pass; the callback records
    /// its own render pass(es) into it.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// View of the current swapchain texture (the present target).
    pub view: &'a wgpu::TextureView,
    /// Current surface configuration (size in physical px, format).
    pub config: &'a wgpu::SurfaceConfiguration,
    /// The window, for input scale factor and redraw scheduling.
    pub window: &'a Window,
}

/// Seam for the overlay / visualizer pass.
///
/// Implemented by the app's overlay today; the future visualizer pass slots in the
/// same way. Invoked once per presented frame, after the clear pass.
pub trait RenderCallback {
    /// Records this frame's overlay/visualizer commands into `frame.encoder`.
    fn render(&mut self, frame: &mut FrameContext<'_>);

    /// Forwards a window event so the callback can update input state (egui).
    ///
    /// Returning `true` means the callback consumed the event (e.g. egui wants the
    /// pointer) — the shell still processes structural events regardless.
    fn on_window_event(&mut self, _window: &Window, _event: &WindowEvent) -> bool {
        false
    }

    /// Whether the shell's initial clear pass should clear to transparent
    /// (`wgpu::Color::TRANSPARENT`) instead of the opaque dark background.
    fn wants_transparent_clear(&self) -> bool {
        false
    }

    /// Whether the callback clears the surface texture itself (e.g. `viz-render`
    /// already clears the swapchain with `LoadOp::Clear`), allowing the shell to
    /// omit its redundant pre-clear render pass.
    fn owns_surface_clear(&self) -> bool {
        false
    }

    /// Desired interval until the next frame. Values below [`MIN_FRAME_INTERVAL`]
    /// are clamped up to the hard 60 Hz cap; larger values let the event loop sleep
    /// longer when showing static guidance or during sustained silence.
    fn desired_frame_interval(&self) -> Duration {
        MIN_FRAME_INTERVAL
    }

    /// Invoked immediately after `queue.submit(...)` so the renderer can recall
    /// `StagingBelt` buffers for reuse on the next frame.
    fn on_post_submit(&mut self) {}
}

/// Live-toggleable window presentation flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindowMode {
    /// Hide window chrome (titlebar / borders); drag canvas to move.
    pub borderless: bool,
    /// Render the active visualizer over a transparent background.
    pub transparent: bool,
    /// Keep the window floating above normal windows.
    pub always_on_top: bool,
}

/// Applies live-toggleable [`WindowMode`] properties to an existing window.
pub fn apply_window_mode(window: &Window, mode: WindowMode) {
    window.set_decorations(!mode.borderless);
    window.set_window_level(if mode.always_on_top {
        WindowLevel::AlwaysOnTop
    } else {
        WindowLevel::Normal
    });
    #[cfg(target_os = "macos")]
    {
        use winit::platform::macos::WindowExtMacOS;
        window.set_has_shadow(!mode.transparent);
    }
}

/// Quick window-size presets exposed in the top-right overlay panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowSizePreset {
    /// Compact 16:9 floating widget (`320 × 180` logical px).
    X1,
    /// 1.5× scale (`480 × 270` logical px).
    X1_5,
    /// 2× scale (`640 × 360` logical px).
    X2,
    /// 4× scale (`1280 × 720` logical px, matching the default launch size).
    X4,
    /// Fill the current display.
    Max,
}

impl WindowSizePreset {
    /// All presets in UI display order.
    pub const ALL: [WindowSizePreset; 5] = [Self::X1, Self::X1_5, Self::X2, Self::X4, Self::Max];

    /// Compact button label for the overlay strip.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::X1 => "1x",
            Self::X1_5 => "1.5x",
            Self::X2 => "2x",
            Self::X4 => "4x",
            Self::Max => "Max",
        }
    }

    /// Hover tooltip text (static to avoid per-frame allocation).
    #[must_use]
    pub const fn hover_text(self) -> &'static str {
        match self {
            Self::X1 => "Resize window to 1x (320×180) — ⌘1",
            Self::X1_5 => "Resize window to 1.5x (480×270) — ⌘2",
            Self::X2 => "Resize window to 2x (640×360) — ⌘3",
            Self::X4 => "Resize window to 4x (1280×720) — ⌘4",
            Self::Max => "Maximize window to fill display — ⌘5",
        }
    }

    /// Target logical `(width, height)` for fixed presets, or `None` for [`Self::Max`].
    #[must_use]
    pub const fn logical_size(self) -> Option<(u32, u32)> {
        match self {
            Self::X1 => Some((PRESET_1X_WIDTH, PRESET_1X_HEIGHT)),
            Self::X1_5 => Some((PRESET_1X_WIDTH * 3 / 2, PRESET_1X_HEIGHT * 3 / 2)),
            Self::X2 => Some((PRESET_1X_WIDTH * 2, PRESET_1X_HEIGHT * 2)),
            Self::X4 => Some((PRESET_1X_WIDTH * 4, PRESET_1X_HEIGHT * 4)),
            Self::Max => None,
        }
    }
}

/// Detects which [`WindowSizePreset`] matches the window's current state.
///
/// Pure function (no windowing I/O) so it is unit-testable.
/// - Returns `Some(Max)` if `is_maximized` is true or if `monitor_logical` matches
///   `inner_logical` within `<= 8.0` logical px on both axes (covering borderless
///   full-display maximize).
/// - Otherwise checks `X1`, `X1_5`, `X2`, `X4` within `<= 4.0` logical px on both axes.
/// - Returns `None` when manually dragged to a custom size.
#[must_use]
pub fn detect_size_preset(
    inner_logical: (f64, f64),
    is_maximized: bool,
    monitor_logical: Option<(f64, f64)>,
) -> Option<WindowSizePreset> {
    if is_maximized {
        return Some(WindowSizePreset::Max);
    }
    if let Some((mon_w, mon_h)) = monitor_logical {
        if (inner_logical.0 - mon_w).abs() <= MAX_MONITOR_TOLERANCE
            && (inner_logical.1 - mon_h).abs() <= MAX_MONITOR_TOLERANCE
        {
            return Some(WindowSizePreset::Max);
        }
    }
    for preset in [
        WindowSizePreset::X1,
        WindowSizePreset::X1_5,
        WindowSizePreset::X2,
        WindowSizePreset::X4,
    ] {
        if let Some((w, h)) = preset.logical_size() {
            if (inner_logical.0 - w as f64).abs() <= SIZE_PRESET_TOLERANCE
                && (inner_logical.1 - h as f64).abs() <= SIZE_PRESET_TOLERANCE
            {
                return Some(preset);
            }
        }
    }
    None
}

/// Applies a [`WindowSizePreset`] to an existing window.
///
/// Fixed presets un-maximize first, clamp the target logical size to the current
/// monitor's logical bounds (so `4x` never overflows a small display), and call
/// [`Window::request_inner_size`]. For [`WindowSizePreset::Max`], borderless windows
/// explicitly position and size to the current monitor because macOS `NSWindow`
/// zoom/maximize is a no-op without `NSWindowStyleMaskTitled`.
pub fn apply_window_size_preset(window: &Window, preset: WindowSizePreset, borderless: bool) {
    match preset.logical_size() {
        Some((target_w, target_h)) => {
            window.set_maximized(false);
            let mut w = target_w as f64;
            let mut h = target_h as f64;
            if let Some(mon) = window.current_monitor() {
                let mon_logical: winit::dpi::LogicalSize<f64> =
                    mon.size().to_logical(mon.scale_factor());
                if mon_logical.width >= 1.0 {
                    w = w.min(mon_logical.width);
                }
                if mon_logical.height >= 1.0 {
                    h = h.min(mon_logical.height);
                }
            }
            let _ = window.request_inner_size(winit::dpi::LogicalSize::new(w, h));
        }
        None => {
            if borderless {
                if let Some(mon) = window.current_monitor() {
                    // Convert to logical points using the *target* monitor's scale
                    // factor: passing `PhysicalSize` directly causes winit on macOS
                    // to divide by the window's previous-monitor scale factor,
                    // doubling dimensions when moving from 1× to a 2× Retina display.
                    let mon_logical: winit::dpi::LogicalSize<f64> =
                        mon.size().to_logical(mon.scale_factor());
                    window.set_outer_position(mon.position());
                    let _ = window.request_inner_size(mon_logical);
                } else {
                    window.set_maximized(true);
                }
            } else {
                window.set_maximized(true);
            }
        }
    }
}

/// One-shot hook run once the GPU context first exists, so the host can build
/// GPU-dependent resources (e.g. the egui renderer needs the surface format).
type ReadyHook<C> = Box<dyn FnOnce(&GpuContext, &Window, &mut C)>;

/// Restored window geometry to apply at creation: outer position + inner size, both
/// in physical pixels. Supplied by the host from persisted state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InitialWindow {
    /// Outer-position x in physical pixels.
    pub x: i32,
    /// Outer-position y in physical pixels.
    pub y: i32,
    /// Inner width in physical pixels.
    pub width: u32,
    /// Inner height in physical pixels.
    pub height: u32,
}

/// Clamps a restored [`InitialWindow`]'s physical dimensions to at least `1×1` and
/// at most the largest available monitor's physical `(width, height)`.
///
/// Pure function so it is unit-testable. Protects against stale or corrupt persisted
/// state (e.g. a cross-monitor double-scaled `6912×2104` entry) opening an oversized
/// off-screen window. When `max_monitor_size` is `None`, only enforces the `>= 1` floor.
#[must_use]
pub fn clamp_initial_window(
    win: InitialWindow,
    max_monitor_size: Option<(u32, u32)>,
) -> InitialWindow {
    let mut width = win.width.max(1);
    let mut height = win.height.max(1);
    if let Some((max_w, max_h)) = max_monitor_size {
        if max_w > 0 {
            width = width.min(max_w);
        }
        if max_h > 0 {
            height = height.min(max_h);
        }
    }
    InitialWindow {
        x: win.x,
        y: win.y,
        width,
        height,
    }
}

/// Decides whether to present this redraw given the hard 60 Hz cap.
///
/// Pure function (no I/O) so it is unit-testable. Returns `true` when at least
/// `min_interval` has elapsed since `last_present` (or there was no previous
/// present). On `false` the caller skips rendering but still re-requests a redraw.
#[must_use]
pub fn should_present(now: Instant, last_present: Option<Instant>, min_interval: Duration) -> bool {
    match last_present {
        None => true,
        Some(prev) => now.duration_since(prev) >= min_interval,
    }
}

/// Result of attempting to acquire, render, and present one swapchain frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderOutcome {
    /// A frame was rendered and presented.
    Presented,
    /// Swapchain reported occlusion; caller should back off without spinning.
    Occluded,
    /// Transient surface error (`Timeout`, `Outdated`, `Lost`, `Validation`); back off.
    RetryLater,
}

/// Computes the next redraw deadline for [`App::about_to_wait`].
///
/// Pure function so scheduling behavior is unit-testable without a window.
#[must_use]
fn next_frame_deadline(
    now: Instant,
    last_present: Option<Instant>,
    next_redraw_at: Option<Instant>,
    desired_interval: Duration,
    wake_immediately: bool,
) -> Instant {
    if wake_immediately {
        return last_present.map(|p| p + MIN_FRAME_INTERVAL).unwrap_or(now);
    }
    let interval = desired_interval.max(MIN_FRAME_INTERVAL);
    let desired = last_present.map(|p| p + interval).unwrap_or(now);
    match next_redraw_at {
        Some(backoff) => backoff.max(desired),
        None => desired,
    }
}

/// The winit application: owns the window, GPU context, render callback, and pacing
/// state. A `&mut` reference is passed to [`winit::event_loop::EventLoop::run_app`].
pub struct App<C: RenderCallback> {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    callback: C,
    /// Instant of the last actually-presented frame (drives 60 Hz pacing).
    last_present: Option<Instant>,
    /// Earliest instant at which the next redraw should be requested (includes
    /// backoff after transient surface errors or `CurrentSurfaceTexture::Occluded`).
    next_redraw_at: Option<Instant>,
    /// True after `window.request_redraw()` until the matching `RedrawRequested` runs.
    redraw_in_flight: bool,
    /// Set by OS `WindowEvent::Occluded(true)`; pauses rendering until unoccluded.
    occluded: bool,
    /// Set when the window's physical size is `0×0` (minimized); pauses rendering.
    minimized: bool,
    /// Set on user input / resize / unocclusion so throttled intervals snap back
    /// immediately to the next 60 Hz slot.
    wake_immediately: bool,
    /// Hook invoked exactly once after the GPU context first exists, so the host can
    /// build GPU-dependent resources (e.g. the egui renderer needs the surface format).
    on_ready: Option<ReadyHook<C>>,
    /// Restored geometry to apply when the window is created; `None` ⇒ use
    /// the platform default size + placement.
    initial_window: Option<InitialWindow>,
    /// Initial window mode (borderless, transparent, always-on-top) applied at creation.
    window_mode: WindowMode,
}

impl<C: RenderCallback> App<C> {
    /// Creates the app with its render callback. The window/GPU are created later in
    /// [`ApplicationHandler::resumed`].
    pub fn new(callback: C) -> Self {
        Self {
            window: None,
            gpu: None,
            callback,
            last_present: None,
            next_redraw_at: None,
            redraw_in_flight: false,
            occluded: false,
            minimized: false,
            wake_immediately: false,
            on_ready: None,
            initial_window: None,
            window_mode: WindowMode::default(),
        }
    }

    /// Registers a one-shot hook run after the GPU context is created (so the host
    /// can build GPU-format-dependent state such as the egui renderer).
    pub fn on_ready(mut self, f: impl FnOnce(&GpuContext, &Window, &mut C) + 'static) -> Self {
        self.on_ready = Some(Box::new(f));
        self
    }

    /// Restores the window's geometry from persisted state. When set, the
    /// window opens at the saved position + size instead of the platform default.
    pub fn with_initial_window(mut self, initial: Option<InitialWindow>) -> Self {
        self.initial_window = initial;
        self
    }

    /// Sets the initial [`WindowMode`] (borderless, transparent, always-on-top) applied
    /// when the window is created.
    pub fn with_window_mode(mut self, mode: WindowMode) -> Self {
        self.window_mode = mode;
        self
    }

    /// Renders one frame: optional shell clear pass into the surface texture, then the
    /// render callback composites on top.
    fn render(&mut self) -> RenderOutcome {
        let (Some(window), Some(gpu)) = (self.window.as_ref(), self.gpu.as_mut()) else {
            return RenderOutcome::RetryLater;
        };

        let surface_texture = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            // Surface needs reconfiguration; do it and retry after backoff.
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let size = window.inner_size();
                gpu.reconfigure(size);
                return RenderOutcome::RetryLater;
            }
            wgpu::CurrentSurfaceTexture::Occluded => return RenderOutcome::Occluded,
            // Transient: skip this frame with backoff.
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Validation => {
                return RenderOutcome::RetryLater;
            }
        };

        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewmusic-frame"),
            });

        // Clear pass FIRST unless the callback owns clearing the surface itself
        // (avoids an empty full-screen clear+store pass every frame on Active views).
        if !self.callback.owns_surface_clear() {
            let clear_color = if self.callback.wants_transparent_clear() {
                wgpu::Color::TRANSPARENT
            } else {
                wgpu::Color {
                    r: 0.02,
                    g: 0.02,
                    b: 0.03,
                    a: 1.0,
                }
            };
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        // Hand the encoder + view to the overlay/visualizer pass.
        {
            let mut frame = FrameContext {
                device: &gpu.device,
                queue: &gpu.queue,
                encoder: &mut encoder,
                view: &view,
                config: &gpu.config,
                window,
            };
            self.callback.render(&mut frame);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        self.callback.on_post_submit();
        // Tell winit we are about to present so it can pace the next redraw.
        window.pre_present_notify();
        surface_texture.present();
        RenderOutcome::Presented
    }
}

impl<C: RenderCallback> ApplicationHandler for App<C> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Recreate only if we have no window yet (resume after suspend keeps it).
        if self.window.is_some() {
            return;
        }

        let mode = self.window_mode;
        // Always enable transparent surface support at creation so Transparent mode
        // can be toggled live at runtime without recreating the window.
        let mut attrs = Window::default_attributes()
            .with_title("ViewMusic")
            .with_transparent(true)
            .with_decorations(!mode.borderless)
            .with_resizable(true)
            .with_window_level(if mode.always_on_top {
                WindowLevel::AlwaysOnTop
            } else {
                WindowLevel::Normal
            });
        #[cfg(target_os = "macos")]
        {
            use winit::platform::macos::WindowAttributesExtMacOS;
            attrs = attrs
                .with_has_shadow(!mode.transparent)
                .with_accepts_first_mouse(true);
        }
        // Restore persisted geometry (physical px) when available, clamped to the
        // largest available monitor so stale/oversized state never opens off-screen.
        let max_monitor_size = event_loop
            .available_monitors()
            .chain(event_loop.primary_monitor())
            .map(|m| m.size())
            .fold(None, |acc: Option<(u32, u32)>, sz| {
                Some(match acc {
                    Some((w, h)) => (w.max(sz.width), h.max(sz.height)),
                    None => (sz.width, sz.height),
                })
            });
        match self.initial_window {
            Some(win) => {
                let win = clamp_initial_window(win, max_monitor_size);
                attrs = attrs
                    .with_inner_size(PhysicalSize::new(win.width, win.height))
                    .with_position(winit::dpi::PhysicalPosition::new(win.x, win.y));
            }
            None => {
                attrs = attrs.with_inner_size(winit::dpi::LogicalSize::new(
                    INITIAL_WIDTH as f64,
                    INITIAL_HEIGHT as f64,
                ));
            }
        }
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        let size = window.inner_size();
        let gpu = GpuContext::new(window.clone(), size);

        // One-shot host hook (e.g. build the egui renderer against the surface format).
        if let Some(f) = self.on_ready.take() {
            f(&gpu, &window, &mut self.callback);
        }

        self.gpu = Some(gpu);
        self.window = Some(window.clone());
        self.redraw_in_flight = true;
        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Let the overlay see input/structural events (skip RedrawRequested, which
        // neither VizApp nor egui_winit consumes).
        if !matches!(event, WindowEvent::RedrawRequested) {
            if let Some(window) = self.window.as_ref() {
                let _consumed = self.callback.on_window_event(window, &event);
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if occluded {
                    self.redraw_in_flight = false;
                } else {
                    self.wake_immediately = true;
                }
            }
            WindowEvent::Resized(size) => {
                self.minimized = size.width == 0 || size.height == 0;
                if self.minimized {
                    self.redraw_in_flight = false;
                } else {
                    if let Some(gpu) = self.gpu.as_mut() {
                        gpu.resize(size);
                    }
                    self.wake_immediately = true;
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                // Physical inner size already reflects the new scale; reconfigure.
                if let (Some(gpu), Some(window)) = (self.gpu.as_mut(), self.window.as_ref()) {
                    gpu.resize(window.inner_size());
                    self.wake_immediately = true;
                }
            }
            WindowEvent::CursorMoved { .. }
            | WindowEvent::CursorEntered { .. }
            | WindowEvent::CursorLeft { .. }
            | WindowEvent::MouseInput { .. }
            | WindowEvent::MouseWheel { .. }
            | WindowEvent::KeyboardInput { .. }
            | WindowEvent::ModifiersChanged(_)
            | WindowEvent::Focused(true) => {
                self.wake_immediately = true;
            }
            WindowEvent::RedrawRequested => {
                self.redraw_in_flight = false;
                if self.occluded || self.minimized {
                    return;
                }
                let now = Instant::now();
                let effective_interval = if self.wake_immediately {
                    MIN_FRAME_INTERVAL
                } else {
                    self.callback
                        .desired_frame_interval()
                        .max(MIN_FRAME_INTERVAL)
                };
                if should_present(now, self.last_present, effective_interval) {
                    self.wake_immediately = false;
                    match self.render() {
                        RenderOutcome::Presented => {
                            self.last_present = Some(now);
                            let next_interval = self
                                .callback
                                .desired_frame_interval()
                                .max(MIN_FRAME_INTERVAL);
                            self.next_redraw_at = Some(now + next_interval);
                        }
                        RenderOutcome::Occluded => {
                            // Slow backoff instead of spinning if OS did not send Occluded(true).
                            self.next_redraw_at = Some(now + Duration::from_millis(100));
                        }
                        RenderOutcome::RetryLater => {
                            self.next_redraw_at = Some(now + MIN_FRAME_INTERVAL);
                        }
                    }
                } else if let Some(prev) = self.last_present {
                    self.next_redraw_at = Some(prev + effective_interval);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(window) = self.window.as_ref() else {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        };
        if self.occluded || self.minimized || self.redraw_in_flight {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }
        let now = Instant::now();
        let deadline = next_frame_deadline(
            now,
            self.last_present,
            self.next_redraw_at,
            self.callback.desired_frame_interval(),
            self.wake_immediately,
        );
        if now >= deadline {
            self.redraw_in_flight = true;
            window.request_redraw();
            event_loop.set_control_flow(ControlFlow::Wait);
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_frame_always_presents() {
        let now = Instant::now();
        assert!(should_present(now, None, MIN_FRAME_INTERVAL));
    }

    #[test]
    fn too_soon_skips() {
        let last = Instant::now();
        // 5 ms later — well under the 15.5 ms budget → skip (120 Hz half-step).
        let now = last + Duration::from_millis(5);
        assert!(!should_present(now, Some(last), MIN_FRAME_INTERVAL));
    }

    #[test]
    fn at_budget_presents() {
        let last = Instant::now();
        let now = last + MIN_FRAME_INTERVAL;
        assert!(should_present(now, Some(last), MIN_FRAME_INTERVAL));
    }

    #[test]
    fn just_over_budget_presents() {
        let last = Instant::now();
        let now = last + MIN_FRAME_INTERVAL + Duration::from_micros(200);
        assert!(should_present(now, Some(last), MIN_FRAME_INTERVAL));
    }

    #[test]
    fn pacing_halves_120hz_to_60hz() {
        // Simulate a 120 Hz display: vsync every ~8.33 ms. Across a sequence we
        // expect alternating present/skip so the effective rate is ~60 Hz.
        let step = Duration::from_micros(8_333);
        let mut last: Option<Instant> = None;
        let start = Instant::now();
        let mut presents = 0;
        for k in 0..120u32 {
            let now = start + step * k;
            if should_present(now, last, MIN_FRAME_INTERVAL) {
                presents += 1;
                last = Some(now);
            }
        }
        // 120 vsyncs over ~1 s should yield ~60 presents (allow a small margin).
        assert!(
            (58..=62).contains(&presents),
            "expected ~60 presents, got {presents}"
        );
    }

    #[test]
    fn size_preset_logical_sizes_and_labels() {
        assert_eq!(WindowSizePreset::X1.logical_size(), Some((320, 180)));
        assert_eq!(WindowSizePreset::X1_5.logical_size(), Some((480, 270)));
        assert_eq!(WindowSizePreset::X2.logical_size(), Some((640, 360)));
        assert_eq!(
            WindowSizePreset::X4.logical_size(),
            Some((INITIAL_WIDTH, INITIAL_HEIGHT))
        );
        assert_eq!(WindowSizePreset::Max.logical_size(), None);

        let labels: Vec<&str> = WindowSizePreset::ALL.iter().map(|p| p.label()).collect();
        assert_eq!(labels, vec!["1x", "1.5x", "2x", "4x", "Max"]);
    }

    #[test]
    fn detect_size_preset_exact_and_within_tolerance() {
        let monitor = Some((1920.0, 1080.0));
        assert_eq!(
            detect_size_preset((320.0, 180.0), false, monitor),
            Some(WindowSizePreset::X1)
        );
        assert_eq!(
            detect_size_preset((480.0, 270.0), false, monitor),
            Some(WindowSizePreset::X1_5)
        );
        assert_eq!(
            detect_size_preset((640.0, 360.0), false, monitor),
            Some(WindowSizePreset::X2)
        );
        assert_eq!(
            detect_size_preset((1280.0, 720.0), false, monitor),
            Some(WindowSizePreset::X4)
        );
        // Within <= 4.0 logical px tolerance on both axes.
        assert_eq!(
            detect_size_preset((324.0, 176.0), false, monitor),
            Some(WindowSizePreset::X1)
        );
        assert_eq!(
            detect_size_preset((636.5, 363.5), false, monitor),
            Some(WindowSizePreset::X2)
        );
        // Just outside tolerance on one axis → custom size.
        assert_eq!(detect_size_preset((324.1, 180.0), false, monitor), None);
        assert_eq!(detect_size_preset((320.0, 184.5), false, monitor), None);
    }

    #[test]
    fn detect_size_preset_custom_returns_none() {
        assert_eq!(
            detect_size_preset((800.0, 500.0), false, Some((1920.0, 1080.0))),
            None
        );
        assert_eq!(detect_size_preset((800.0, 500.0), false, None), None);
    }

    #[test]
    fn detect_size_preset_maximized_and_borderless_monitor_fill() {
        // Explicit maximized flag always reports Max regardless of inner size.
        assert_eq!(
            detect_size_preset((1280.0, 720.0), true, Some((1920.0, 1080.0))),
            Some(WindowSizePreset::Max)
        );
        assert_eq!(
            detect_size_preset((1000.0, 600.0), true, None),
            Some(WindowSizePreset::Max)
        );
        // Borderless monitor fill (is_maximized == false) within <= 8.0 logical px.
        assert_eq!(
            detect_size_preset((1920.0, 1080.0), false, Some((1920.0, 1080.0))),
            Some(WindowSizePreset::Max)
        );
        assert_eq!(
            detect_size_preset((1912.0, 1072.0), false, Some((1920.0, 1080.0))),
            Some(WindowSizePreset::Max)
        );
        // Outside 8.0 logical px tolerance from monitor size → None.
        assert_eq!(
            detect_size_preset((1911.0, 1080.0), false, Some((1920.0, 1080.0))),
            None
        );
    }

    #[test]
    fn clamp_initial_window_caps_oversized_state_to_monitor_bounds() {
        // Stale double-scaled 6912×2104 state gets clamped to a 3456×2234 Retina monitor.
        let oversized = InitialWindow {
            x: 0,
            y: 130,
            width: 6912,
            height: 2104,
        };
        assert_eq!(
            clamp_initial_window(oversized, Some((3456, 2234))),
            InitialWindow {
                x: 0,
                y: 130,
                width: 3456,
                height: 2104,
            }
        );

        // Normal size within bounds passes through unchanged.
        let normal = InitialWindow {
            x: 100,
            y: 80,
            width: 1280,
            height: 720,
        };
        assert_eq!(clamp_initial_window(normal, Some((3456, 2234))), normal);

        // Zero dimensions are floored to 1; missing monitor info keeps valid size.
        let zero = InitialWindow {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        };
        assert_eq!(
            clamp_initial_window(zero, None),
            InitialWindow {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            }
        );
        assert_eq!(clamp_initial_window(normal, None), normal);
    }

    #[test]
    fn next_frame_deadline_respects_pacing_backoff_and_wake() {
        let now = Instant::now();
        // First frame with no prior present is due immediately.
        assert_eq!(
            next_frame_deadline(now, None, None, MIN_FRAME_INTERVAL, false),
            now
        );
        // Active 60 Hz frame schedules last_present + MIN_FRAME_INTERVAL.
        assert_eq!(
            next_frame_deadline(now, Some(now), None, MIN_FRAME_INTERVAL, false),
            now + MIN_FRAME_INTERVAL
        );
        // Throttled 20 Hz silence interval waits 50 ms unless input wakes it.
        let silent = Duration::from_millis(50);
        assert_eq!(
            next_frame_deadline(now, Some(now), None, silent, false),
            now + silent
        );
        assert_eq!(
            next_frame_deadline(now, Some(now), None, silent, true),
            now + MIN_FRAME_INTERVAL
        );
        // Backoff after Occluded/RetryLater overrides shorter desired interval.
        let backoff = now + Duration::from_millis(100);
        assert_eq!(
            next_frame_deadline(now, Some(now), Some(backoff), MIN_FRAME_INTERVAL, false),
            backoff
        );
    }
}
