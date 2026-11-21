//! Window placement remembered per monitor configuration.
//!
//! A *monitor configuration* is the set of displays connected right now: the laptop
//! panel alone, the laptop plus one external display, the external display alone
//! with the lid closed, and so on. Each configuration gets its own
//! [`WindowPlacement`] (window geometry plus the chrome flags in [`WindowMode`]), so
//! docking and undocking puts the window back where it was the last time that set of
//! displays was connected. Everything else the app persists (selected artifact,
//! setting values, overlay collapse) is global and lives in
//! [`crate::persist::AppState`].
//!
//! ## Desktop units
//! All geometry here ([`WindowGeometry`], [`MonitorDescriptor`]) is in *desktop
//! units*: the platform's global window coordinate space. That is logical points on
//! macOS (AppKit's coordinate space, where every display shares one point grid) and
//! physical pixels on Windows and elsewhere (the virtual-screen space). winit's
//! "physical" coordinates are not usable as a global space on macOS: winit scales
//! each monitor's rect by *that* monitor's scale factor, so a 2× Retina panel and a
//! 1× external display report overlapping physical rects. The conversion lives in
//! [`crate::window`].
//!
//! ## Config key
//! [`MonitorConfig::key`] fingerprints the connected displays: each display is
//! described as `"<name> <width>x<height>@<x>,<y>"` (desktop units), the descriptors
//! are sorted by position, and joined with `" + "`, e.g.
//! `"Monitor #41003 1728x1117@0,0 + Monitor #16642 2560x1440@1728,-200"`. It is
//! deterministic (independent of enumeration order) and readable in the JSON.
//!
//! * **Name** is whatever winit reports. On macOS winit 0.30 returns
//!   `"Monitor #<CGDisplayModelNumber>"` (not the marketing name such as
//!   "DELL U2720Q"): stable across reconnects and ports, but shared by identical
//!   models. On Windows it is the GDI device name (`\\.\DISPLAY1`), which names the
//!   output, not the panel, and can change when cables move between ports.
//! * **Position** tells two identical models apart (two displays never share an
//!   origin unless mirrored) and pins the arrangement. Saved window coordinates are
//!   only meaningful in the arrangement they were saved in, so rearranging displays
//!   (or moving the macOS menu bar to another display) is treated as a new
//!   configuration on purpose.
//! * **Size** (desktop units) is included for the same reason: a saved rect from a
//!   1920×1080 desktop may not fit a 1280×800 one.
//! * **Scale factor is excluded.** It does not identify the hardware, and saved
//!   geometry stays valid across scale changes: on Windows desktop units are physical
//!   pixels, which a DPI change does not alter; on macOS a scaling change already
//!   changes the display's size in points, so the key changes anyway. Leaving it out
//!   also keeps fractional floats out of the key.
//!
//! ## `window_state.json`
//! [`WindowProfiles`] is the on-disk document (config dir, see [`crate::persist`]):
//!
//! ```json
//! {
//!   "last_config": "Monitor #41003 1728x1117@0,0",
//!   "use_counter": 7,
//!   "profiles": {
//!     "Monitor #41003 1728x1117@0,0": {
//!       "x": 200, "y": 120, "width": 640, "height": 360,
//!       "borderless": true, "transparent": false, "always_on_top": true,
//!       "last_used": 7
//!     }
//!   }
//! }
//! ```
//!
//! `use_counter` is bumped each time a configuration becomes current or gets a
//! profile, and the profile's `last_used` takes the new value. Every save keeps only
//! the 16 (`MAX_PROFILES`) most recently used profiles, never dropping the current one.
//! Files from before this field default both to `0` (ties are dropped in key order).
//!
//! Older builds wrote a single `{x, y, width, height}` object here (winit physical
//! pixels) and kept `borderless` / `transparent` / `always_on_top` in
//! `app_state.json`. [`WindowProfiles::load`] recognizes that format and the first
//! launch adopts it as the profile for the monitors connected at that launch (the
//! best available guess for where it was saved), then rewrites the file in the new
//! format immediately so the old `app_state.json` flags are never needed again.
//!
//! ## Launch
//! [`PlacementTracker::start`] picks the profile for the launch configuration. With
//! no profile for it, the window opens at the platform default size and placement
//! (macOS centers it on the main display) with the chrome flags of the most recently
//! used profile. Restored geometry is always passed through [`MonitorConfig::fit`],
//! so a rect saved on a display layout that has since changed still lands visibly on
//! a connected display.
//!
//! If winit reports no display at all at launch (the host falls back to the primary
//! monitor first), the window opens at the platform default with the last-used
//! chrome, nothing is recorded under the empty configuration, and old-format state
//! stays pending. The first check that sees displays then switches to them as on a
//! hotplug, adopting the old-format state at that point.
//!
//! ## Hotplug
//! winit has no "displays changed" event. [`PlacementTracker::poll`] runs from the
//! render loop; on most frames it is a single `Instant` comparison. Every
//! [`MONITOR_CHECK_INTERVAL`] (or right away after [`PlacementTracker::request_check`],
//! which the host calls on `ScaleFactorChanged`) it enumerates the displays and reads
//! the window's geometry:
//!
//! * Same configuration: the geometry becomes a *candidate*, and the previous
//!   check's candidate is committed to the current profile (saved on the
//!   [`WINDOW_SAVE_DEBOUNCE`] debounce). A reading is thus committed only once the
//!   next check confirms the displays did not change in between, which covers an OS
//!   that moves the window (and sends `ScaleFactorChanged`, triggering an early
//!   check) before the display list reflects the change. The cost is about one more
//!   check interval (~1 s) before a move is saved. Quit
//!   ([`PlacementTracker::finish`]) commits right away if the displays are still the
//!   current configuration and no change is pending.
//! * Different configuration: neither this geometry nor the pending candidate is
//!   committed. When a display goes away the OS moves and resizes the window itself,
//!   and that rect must not overwrite the old configuration's profile, which keeps
//!   the geometry from its last confirmed check. The new configuration becomes
//!   current only after it has been observed for [`MONITOR_CONFIG_SETTLE`], so a
//!   transient reading taken mid-reconfiguration (a display enumerated before its
//!   final mode or arrangement is set; an empty display list is ignored outright,
//!   apart from dropping the candidate) does not create a bogus profile. If the new
//!   configuration has a profile, the host applies it (geometry and chrome);
//!   otherwise the window stays as it is and its current state becomes the new
//!   configuration's profile. If the user toggled chrome while the change was
//!   pending, the switch keeps the window's chrome and writes it into the new
//!   profile instead of applying the saved chrome (the toggle is also recorded in the
//!   old profile, which was still current when it happened).
//! * Back to the current configuration before the change settled (a display blip):
//!   the OS may have moved the window meanwhile, so nothing is committed and the
//!   host re-applies the current profile's saved placement.
//!
//! Limits: a move the user made in the last one to two check intervals before a
//! display change is lost (it was never confirmed); the switch happens one to two
//! seconds after the change; a blip that starts and ends between two checks cannot
//! be detected, so an OS move during it is committed like a user move; and detection
//! pauses while the window is occluded or minimized (the render loop does not run),
//! catching up on the next frame.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::persist::{
    legacy_window_mode, read_json, should_save_window, write_json, WINDOW_SAVE_DEBOUNCE,
};
use crate::window::WindowMode;

/// File name of the per-monitor-config window document in the config dir.
const WINDOW_STATE_FILE: &str = "window_state.json";

/// How often [`PlacementTracker::poll`] re-enumerates the displays. Enumeration
/// allocates and queries the window server, so it runs at this low rate instead of
/// every frame.
pub const MONITOR_CHECK_INTERVAL: Duration = Duration::from_secs(1);

/// How long a changed monitor configuration must be observed before the window
/// switches to it (see the module docs on transient readings).
pub const MONITOR_CONFIG_SETTLE: Duration = Duration::from_secs(1);

/// Most profiles kept in `window_state.json`; on save the least recently used ones
/// beyond this are dropped (never the current configuration's).
const MAX_PROFILES: usize = 16;

/// Minimum overlap, per axis in desktop units, for a window to count as visible on a
/// display (enough to grab and drag it back). Windows smaller than this only need to
/// overlap by their own size.
const MIN_VISIBLE_EXTENT: u32 = 64;

/// A window's outer top-left position and inner (content) size, in desktop units
/// (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowGeometry {
    /// Outer (frame) left edge.
    pub x: i32,
    /// Outer (frame) top edge.
    pub y: i32,
    /// Inner (content) width.
    pub width: u32,
    /// Inner (content) height.
    pub height: u32,
}

/// One connected display.
#[derive(Clone, Debug, PartialEq)]
pub struct MonitorDescriptor {
    /// The display name winit reports (see the module docs for what it contains per
    /// platform).
    pub name: String,
    /// Left edge in desktop units.
    pub x: i32,
    /// Top edge in desktop units.
    pub y: i32,
    /// Width in desktop units.
    pub width: u32,
    /// Height in desktop units.
    pub height: u32,
    /// winit physical pixels per desktop unit on this display: its scale factor on
    /// macOS, `1.0` elsewhere. Not part of the config key; used only to convert the
    /// physical-pixel geometry of old `window_state.json` files.
    pub physical_px_per_unit: f64,
}

/// The set of currently connected displays, identified by a deterministic key.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MonitorConfig {
    /// See [`MonitorConfig::key`].
    key: String,
    /// The displays, sorted in key order.
    monitors: Vec<MonitorDescriptor>,
}

impl MonitorConfig {
    /// Builds the configuration for `monitors`, in any order.
    pub fn new(mut monitors: Vec<MonitorDescriptor>) -> Self {
        monitors.sort_by(|a, b| {
            (a.x, a.y, &a.name, a.width, a.height).cmp(&(b.x, b.y, &b.name, b.width, b.height))
        });
        let mut key = String::new();
        for (i, m) in monitors.iter().enumerate() {
            if i > 0 {
                key.push_str(" + ");
            }
            let _ = write!(key, "{} {}x{}@{},{}", m.name, m.width, m.height, m.x, m.y);
        }
        Self { key, monitors }
    }

    /// The configuration's fingerprint, used as the profile key in
    /// `window_state.json`. Format and rationale are in the module docs.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Whether no display was reported (seen transiently while displays reconfigure,
    /// and on platforms that cannot enumerate displays).
    pub fn is_empty(&self) -> bool {
        self.monitors.is_empty()
    }

    /// The primary display: the one at the desktop origin (macOS and Windows both
    /// place the primary display there), else the first in key order.
    fn primary(&self) -> Option<&MonitorDescriptor> {
        self.monitors
            .iter()
            .find(|m| m.x == 0 && m.y == 0)
            .or_else(|| self.monitors.first())
    }

    /// Returns `geometry` adjusted so the window is visible on a connected display.
    ///
    /// A window that overlaps some display by at least 64 desktop units per axis (or
    /// its own size, if smaller) keeps its position, so windows straddling two
    /// displays stay put; its size is capped to the display it overlaps most. Any
    /// other window is capped to the primary display's size and centered on it. Sizes
    /// are at least `1×1`. With no displays known, only the size floor applies.
    #[must_use]
    pub fn fit(&self, geometry: WindowGeometry) -> WindowGeometry {
        let rect = WindowGeometry {
            width: geometry.width.max(1),
            height: geometry.height.max(1),
            ..geometry
        };
        let best = self
            .monitors
            .iter()
            .filter(|m| is_visible_on(rect, m))
            .max_by_key(|m| {
                let (w, h) = overlap(rect, m);
                u64::from(w) * u64::from(h)
            });
        let Some(target) = best.or_else(|| self.primary()) else {
            return rect;
        };
        let sized = WindowGeometry {
            width: rect.width.min(target.width.max(1)),
            height: rect.height.min(target.height.max(1)),
            ..rect
        };
        // Capping the size shrinks the window toward its top-left corner, which can
        // pull it off the display it overlapped; recenter in that case too.
        if best.is_some() && self.monitors.iter().any(|m| is_visible_on(sized, m)) {
            return sized;
        }
        WindowGeometry {
            x: target.x + (target.width.saturating_sub(sized.width) / 2) as i32,
            y: target.y + (target.height.saturating_sub(sized.height) / 2) as i32,
            ..sized
        }
    }
}

/// Per-axis overlap `(width, height)` between `rect` and display `m`, in desktop
/// units (zero when disjoint).
fn overlap(rect: WindowGeometry, m: &MonitorDescriptor) -> (u32, u32) {
    let axis = |a0: i32, a_len: u32, b0: i32, b_len: u32| -> u32 {
        let start = i64::from(a0).max(i64::from(b0));
        let end = (i64::from(a0) + i64::from(a_len)).min(i64::from(b0) + i64::from(b_len));
        (end - start).clamp(0, i64::from(u32::MAX)) as u32
    };
    (
        axis(rect.x, rect.width, m.x, m.width),
        axis(rect.y, rect.height, m.y, m.height),
    )
}

/// Whether `rect` overlaps display `m` enough to be found and dragged (see
/// [`MIN_VISIBLE_EXTENT`]).
fn is_visible_on(rect: WindowGeometry, m: &MonitorDescriptor) -> bool {
    let (w, h) = overlap(rect, m);
    w >= MIN_VISIBLE_EXTENT.min(rect.width) && h >= MIN_VISIBLE_EXTENT.min(rect.height)
}

/// Where and how the window is shown for one monitor configuration: one profile in
/// `window_state.json`, and what the host applies at launch or on a config switch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowPlacement {
    /// Window geometry, or `None` when unknown (open at the platform default size and
    /// placement / leave the window where it is).
    #[serde(flatten)]
    pub geometry: Option<WindowGeometry>,
    /// Chrome flags (borderless, transparent, always on top).
    #[serde(flatten)]
    pub mode: WindowMode,
}

/// One entry of `window_state.json`'s `profiles`: a [`WindowPlacement`] plus when it
/// was last used, which decides what [`WindowProfiles::prune`] drops.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct WindowProfile {
    /// Geometry and chrome for this configuration.
    #[serde(flatten)]
    placement: WindowPlacement,
    /// [`WindowProfiles::use_counter`] value from when this configuration was last
    /// made current (or its profile created). `0` in files from before pruning.
    #[serde(default)]
    last_used: u64,
}

/// The `window_state.json` document: one profile per monitor config key.
/// Schema and migration are described in the module docs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowProfiles {
    /// Key of the configuration the window was most recently placed for. Its profile
    /// supplies the chrome flags when launching on a configuration with no profile.
    last_config: Option<String>,
    /// Monotonic counter behind [`WindowProfile::last_used`]; bumped each time a
    /// configuration becomes current or gets a profile.
    use_counter: u64,
    /// Profile per monitor config key, at most [`MAX_PROFILES`] after a save.
    profiles: BTreeMap<String, WindowProfile>,
    /// State from the pre-per-config format, adopted by [`PlacementTracker::start`]
    /// once the launch configuration is known. Never serialized.
    #[serde(skip)]
    legacy: Option<LegacyWindowSetup>,
}

/// Window state saved by builds before per-config profiles.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LegacyWindowSetup {
    /// The old `window_state.json` geometry, in winit physical pixels.
    physical_geometry: Option<WindowGeometry>,
    /// The chrome flags old builds kept in `app_state.json`.
    mode: WindowMode,
}

/// Both formats `window_state.json` has had. `Legacy` is tried first because the
/// current format (all fields defaulted) would accept any object.
#[derive(Deserialize)]
#[serde(untagged)]
enum StoredWindowState {
    /// Pre-per-config `{x, y, width, height}` in winit physical pixels.
    Legacy(WindowGeometry),
    /// The current document.
    Profiles(WindowProfiles),
}

impl WindowProfiles {
    /// Loads `window_state.json`, recognizing the old single-geometry format and the
    /// chrome flags old builds kept in `app_state.json`. A missing or corrupt file
    /// yields an empty document (persistence is best-effort).
    pub fn load() -> Self {
        Self::from_stored(read_json(WINDOW_STATE_FILE), legacy_window_mode)
    }

    /// Saves `window_state.json` (best-effort: a write failure is logged, not fatal).
    pub fn save(&self) {
        write_json(WINDOW_STATE_FILE, self);
    }

    /// Builds the document from what was on disk. `legacy_mode` reads the chrome
    /// flags from the old `app_state.json`; it is consulted only when the file is not
    /// in the current format.
    fn from_stored(
        stored: Option<StoredWindowState>,
        legacy_mode: impl FnOnce() -> WindowMode,
    ) -> Self {
        let (physical_geometry, mode) = match stored {
            Some(StoredWindowState::Profiles(profiles)) => return profiles,
            Some(StoredWindowState::Legacy(geometry)) => (Some(geometry), legacy_mode()),
            None => (None, legacy_mode()),
        };
        if physical_geometry.is_none() && mode == WindowMode::default() {
            // Fresh install: nothing to carry over.
            return Self::default();
        }
        Self {
            legacy: Some(LegacyWindowSetup {
                physical_geometry,
                mode,
            }),
            ..Self::default()
        }
    }

    /// Turns pending legacy state into the profile for `monitors`. Returns whether
    /// anything was adopted (the caller then saves, so the old format is not needed
    /// again). With no displays known the legacy state stays pending: there is no
    /// configuration to file it under and no scale to convert its pixels with.
    fn adopt_legacy(&mut self, monitors: &MonitorConfig) -> bool {
        if monitors.is_empty() {
            return false;
        }
        let Some(legacy) = self.legacy.take() else {
            return false;
        };
        // Old builds handed the physical rect to winit at creation, which (on macOS)
        // converts it with the main display's scale factor; do the same.
        let px_per_unit = monitors
            .primary()
            .map_or(1.0, |m| m.physical_px_per_unit)
            .max(f64::MIN_POSITIVE);
        let to_units = |v: f64| (v / px_per_unit).round();
        let geometry = legacy.physical_geometry.map(|g| WindowGeometry {
            x: to_units(f64::from(g.x)) as i32,
            y: to_units(f64::from(g.y)) as i32,
            width: to_units(f64::from(g.width)) as u32,
            height: to_units(f64::from(g.height)) as u32,
        });
        self.insert_profile(
            monitors.key().to_owned(),
            WindowPlacement {
                geometry,
                mode: legacy.mode,
            },
        );
        self.last_config = Some(monitors.key().to_owned());
        true
    }

    /// The placement to open with on config `key`: its profile if one exists, else
    /// unknown geometry with the chrome flags still pending from the old format or,
    /// failing that, of the most recently used profile. An empty `key` (no displays
    /// known) never has a profile.
    fn launch_placement(&self, key: &str) -> WindowPlacement {
        if let Some(saved) = self.profiles.get(key).filter(|_| !key.is_empty()) {
            return saved.placement;
        }
        let mode = match self.legacy {
            Some(legacy) => legacy.mode,
            None => self
                .last_config
                .as_ref()
                .and_then(|last| self.profiles.get(last))
                .map(|last| last.placement.mode)
                .unwrap_or_default(),
        };
        WindowPlacement {
            geometry: None,
            mode,
        }
    }

    /// Adds (or replaces) the profile for `key`, marked as used now.
    fn insert_profile(&mut self, key: String, placement: WindowPlacement) {
        self.use_counter += 1;
        self.profiles.insert(
            key,
            WindowProfile {
                placement,
                last_used: self.use_counter,
            },
        );
    }

    /// Records that configuration `key` was just made current (a no-op on the
    /// profile when it has none yet; [`Self::insert_profile`] marks it when created).
    fn mark_used(&mut self, key: &str) {
        self.use_counter += 1;
        if let Some(profile) = self.profiles.get_mut(key) {
            profile.last_used = self.use_counter;
        }
    }

    /// Drops the least recently used profiles beyond [`MAX_PROFILES`], never the one
    /// for [`Self::last_config`] (the current configuration whenever displays are
    /// known). Ties (files from before pruning) go in key order.
    fn prune(&mut self) {
        while self.profiles.len() > MAX_PROFILES {
            let oldest = self
                .profiles
                .iter()
                .filter(|(key, _)| self.last_config.as_ref() != Some(*key))
                .min_by_key(|(_, profile)| profile.last_used)
                .map(|(key, _)| key.clone());
            let Some(oldest) = oldest else {
                return;
            };
            self.profiles.remove(&oldest);
        }
    }
}

/// What [`PlacementTracker::poll`] observes about the live window at one instant.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowSnapshot {
    /// The displays connected now.
    pub monitors: MonitorConfig,
    /// The window's geometry, or `None` when it has none worth saving (minimized).
    pub geometry: Option<WindowGeometry>,
}

/// Persists a [`WindowProfiles`] document (the host writes `window_state.json`).
type SaveHook = Box<dyn FnMut(&WindowProfiles)>;

/// Keeps the window's per-monitor-config profiles up to date while the app runs and
/// tells the host when a display change calls for a different placement (see the
/// module docs for the policy).
///
/// Pure state machine: callers pass in the clock and what the window reports, and
/// apply the returned [`WindowPlacement`]s themselves. [`PlacementTracker::default`]
/// keeps everything in memory (tests).
#[derive(Default)]
pub struct PlacementTracker {
    /// The profiles document, kept in memory and written through `save`.
    profiles: WindowProfiles,
    /// The configuration the window is currently placed for; `None` before
    /// [`PlacementTracker::start`]. Empty when no display was known at launch, until
    /// the first check that sees displays switches to them.
    monitors: Option<MonitorConfig>,
    /// A different configuration seen by a check, with the instant it was first seen,
    /// waiting out [`MONITOR_CONFIG_SETTLE`].
    pending: Option<(MonitorConfig, Instant)>,
    /// Geometry from the latest check that saw the current configuration, committed
    /// only once the next check sees that configuration too.
    candidate: Option<WindowGeometry>,
    /// Whether the user changed the chrome while `pending` was set (or while no
    /// display was known). The next switch then keeps the window's chrome instead of
    /// applying the saved one, so the toggle is not undone.
    mode_changed_while_settling: bool,
    /// Earliest instant of the next display check; `None` ⇒ due now.
    next_check: Option<Instant>,
    /// Instant of the last save (debounce).
    last_save: Option<Instant>,
    /// Whether `profiles` changed since the last save.
    unsaved: bool,
    /// Writes the document; `None` keeps it in memory only.
    save: Option<SaveHook>,
}

impl PlacementTracker {
    /// Creates a tracker over `profiles` that persists through `save`.
    pub fn new(profiles: WindowProfiles, save: impl FnMut(&WindowProfiles) + 'static) -> Self {
        Self {
            profiles,
            save: Some(Box::new(save)),
            ..Self::default()
        }
    }

    /// Resolves the placement to create the window with on `monitors` (the displays
    /// connected at launch) and makes that configuration current. Restored geometry
    /// is already [`MonitorConfig::fit`] to `monitors`. Adopts and saves pre-per-config
    /// state on the first launch after an upgrade.
    ///
    /// With no display known (`monitors` empty) the window opens at the platform
    /// default with the last-used (or pending legacy) chrome, nothing is recorded, and
    /// the first check that sees displays switches to them as on a hotplug, adopting
    /// any pending legacy state then.
    pub fn start(&mut self, monitors: MonitorConfig, now: Instant) -> WindowPlacement {
        let migrated = self.profiles.adopt_legacy(&monitors);
        let saved = self.profiles.launch_placement(monitors.key());
        let placement = WindowPlacement {
            geometry: saved.geometry.map(|g| monitors.fit(g)),
            mode: saved.mode,
        };
        tracing::info!(
            target: "viz_app::placement",
            config = monitors.key(),
            restored_geometry = placement.geometry.is_some(),
            migrated,
            "window placement for launch monitor config"
        );
        if !monitors.is_empty() {
            self.profiles.last_config = Some(monitors.key().to_owned());
            self.profiles.mark_used(monitors.key());
            self.unsaved = true;
        }
        self.monitors = Some(monitors);
        self.pending = None;
        self.candidate = None;
        self.mode_changed_while_settling = false;
        self.next_check = None;
        if migrated {
            self.save_now(now);
        }
        placement
    }

    /// Records chrome flags the user just changed, in the current configuration's
    /// profile only, and saves right away. A no-op before
    /// [`PlacementTracker::start`]. During a pending configuration change (or before
    /// any display is known) it also makes the next switch keep this chrome.
    pub fn record_mode(&mut self, mode: WindowMode, now: Instant) {
        if self.monitors.is_none() {
            return;
        }
        if self.pending.is_some() || self.current_key().is_none() {
            self.mode_changed_while_settling = true;
        }
        let Some(key) = known_key(&self.monitors) else {
            return;
        };
        match self.profiles.profiles.get_mut(key) {
            Some(profile) if profile.placement.mode == mode => return,
            Some(profile) => profile.placement.mode = mode,
            None => {
                let key = key.to_owned();
                self.profiles.insert_profile(
                    key,
                    WindowPlacement {
                        geometry: None,
                        mode,
                    },
                );
            }
        }
        self.save_now(now);
    }

    /// Makes the next [`PlacementTracker::poll`] check the displays regardless of the
    /// interval (call on hints of a display change such as `ScaleFactorChanged`).
    pub fn request_check(&mut self) {
        self.next_check = None;
    }

    /// Drives the tracker from the render loop. When a display check is due it calls
    /// `snapshot` (the only part that allocates or queries the window server), updates
    /// the profiles, and returns the placement to apply if the window just switched to
    /// a configuration that has a saved profile, or came back from a display blip.
    /// Also flushes unsaved changes once the save debounce allows. `mode` is the
    /// window's current chrome, used to start a profile for a configuration seen for
    /// the first time.
    pub fn poll(
        &mut self,
        now: Instant,
        mode: WindowMode,
        snapshot: impl FnOnce() -> WindowSnapshot,
    ) -> Option<WindowPlacement> {
        let due = self.monitors.is_some() && self.next_check.is_none_or(|at| now >= at);
        let to_apply = if due {
            self.next_check = Some(now + MONITOR_CHECK_INTERVAL);
            self.check(now, mode, snapshot())
        } else {
            None
        };
        if self.unsaved && should_save_window(now, self.last_save, WINDOW_SAVE_DEBOUNCE) {
            self.save_now(now);
        }
        to_apply
    }

    /// Final update on quit: if the displays are still the current configuration (and
    /// no change is pending), commits `snapshot`'s geometry right away, without
    /// waiting for a confirming check. Then saves any unsaved change regardless of the
    /// debounce. Never switches configuration.
    pub fn finish(&mut self, now: Instant, mode: WindowMode, snapshot: WindowSnapshot) {
        if self.pending.is_none() && self.is_current(&snapshot.monitors) {
            if let Some(geometry) = snapshot.geometry.or(self.candidate) {
                self.commit_geometry(geometry, mode);
            }
        }
        if self.unsaved {
            self.save_now(now);
        }
    }

    /// One display check (see the module docs for the policy).
    fn check(
        &mut self,
        now: Instant,
        mode: WindowMode,
        snapshot: WindowSnapshot,
    ) -> Option<WindowPlacement> {
        if snapshot.monitors.is_empty() {
            // Displays mid-reconfiguration: neither commit nor count it as a config,
            // and do not let the previous reading be confirmed across it.
            self.candidate = None;
            return None;
        }
        if self.is_current(&snapshot.monitors) {
            if self.pending.take().is_some() {
                // Back from a short blip: the OS may have moved the window while the
                // other configuration was live, so put it back instead of committing.
                self.candidate = None;
                self.mode_changed_while_settling = false;
                return self.current_saved_placement(&snapshot.monitors);
            }
            // Two consecutive checks on this configuration: the earlier reading is
            // the user's placement, not something the OS did on a display change.
            if let Some(confirmed) = self.candidate.take() {
                self.commit_geometry(confirmed, mode);
            }
            self.candidate = snapshot.geometry;
            return None;
        }
        // The displays changed, so this geometry (and the previous, unconfirmed one)
        // may be the OS's doing: never commit it to the current (old) profile.
        self.candidate = None;
        match &self.pending {
            Some((candidate, since)) if candidate.key() == snapshot.monitors.key() => {
                if now.saturating_duration_since(*since) >= MONITOR_CONFIG_SETTLE {
                    self.switch_to(now, mode, snapshot)
                } else {
                    None
                }
            }
            _ => {
                self.pending = Some((snapshot.monitors, now));
                None
            }
        }
    }

    /// The current configuration's saved placement, fitted to `monitors`, when it has
    /// saved geometry to restore.
    fn current_saved_placement(&self, monitors: &MonitorConfig) -> Option<WindowPlacement> {
        let saved = self.profiles.profiles.get(self.current_key()?)?.placement;
        Some(WindowPlacement {
            geometry: Some(monitors.fit(saved.geometry?)),
            mode: saved.mode,
        })
    }

    /// Makes `snapshot.monitors` the current configuration. Returns its saved
    /// placement (fitted to its displays) to apply, or `None` after starting a new
    /// profile from the window as it is now. If the user changed the chrome while the
    /// switch was pending, `mode` is kept and stored in the profile instead.
    fn switch_to(
        &mut self,
        now: Instant,
        mode: WindowMode,
        snapshot: WindowSnapshot,
    ) -> Option<WindowPlacement> {
        let key = snapshot.monitors.key().to_owned();
        // First displays seen after a launch without any: adopt the old format now.
        let migrated = self.profiles.adopt_legacy(&snapshot.monitors);
        let keep_mode = std::mem::take(&mut self.mode_changed_while_settling);
        let to_apply = match self.profiles.profiles.get_mut(&key) {
            Some(saved) => {
                if keep_mode {
                    saved.placement.mode = mode;
                }
                Some(WindowPlacement {
                    geometry: saved.placement.geometry.map(|g| snapshot.monitors.fit(g)),
                    mode: saved.placement.mode,
                })
            }
            None => {
                self.profiles.insert_profile(
                    key.clone(),
                    WindowPlacement {
                        geometry: snapshot.geometry,
                        mode,
                    },
                );
                None
            }
        };
        tracing::info!(
            target: "viz_app::placement",
            from = self.monitors.as_ref().map_or("", MonitorConfig::key),
            to = key.as_str(),
            restored_profile = to_apply.is_some(),
            migrated,
            "monitor config changed"
        );
        self.profiles.mark_used(&key);
        self.profiles.last_config = Some(key);
        self.monitors = Some(snapshot.monitors);
        self.pending = None;
        // Writes the old profile as of its last confirmed check plus the new one.
        self.save_now(now);
        to_apply
    }

    /// Key of the current configuration, or `None` before
    /// [`PlacementTracker::start`] or while no display is known.
    fn current_key(&self) -> Option<&str> {
        known_key(&self.monitors)
    }

    /// Whether `monitors` is the (known) current configuration.
    fn is_current(&self, monitors: &MonitorConfig) -> bool {
        self.current_key() == Some(monitors.key())
    }

    /// Stores `geometry` in the current configuration's profile, creating the profile
    /// with `mode` if it does not exist yet.
    fn commit_geometry(&mut self, geometry: WindowGeometry, mode: WindowMode) {
        let Some(key) = known_key(&self.monitors) else {
            return;
        };
        match self.profiles.profiles.get_mut(key) {
            Some(profile) if profile.placement.geometry == Some(geometry) => return,
            Some(profile) => profile.placement.geometry = Some(geometry),
            None => {
                let key = key.to_owned();
                self.profiles.insert_profile(
                    key,
                    WindowPlacement {
                        geometry: Some(geometry),
                        mode,
                    },
                );
            }
        }
        self.unsaved = true;
    }

    /// Prunes old profiles, writes the document through the save hook, and resets
    /// the debounce.
    fn save_now(&mut self, now: Instant) {
        self.profiles.prune();
        if let Some(save) = self.save.as_mut() {
            save(&self.profiles);
        }
        self.last_save = Some(now);
        self.unsaved = false;
    }
}

/// Key of `monitors` when it is a known (non-empty) configuration. A free function
/// so callers can borrow the tracker's `monitors` and `profiles` fields separately.
fn known_key(monitors: &Option<MonitorConfig>) -> Option<&str> {
    monitors
        .as_ref()
        .filter(|current| !current.is_empty())
        .map(MonitorConfig::key)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;

    fn monitor(name: &str, x: i32, y: i32, width: u32, height: u32) -> MonitorDescriptor {
        MonitorDescriptor {
            name: name.to_owned(),
            x,
            y,
            width,
            height,
            physical_px_per_unit: 1.0,
        }
    }

    fn geometry(x: i32, y: i32, width: u32, height: u32) -> WindowGeometry {
        WindowGeometry {
            x,
            y,
            width,
            height,
        }
    }

    const BORDERLESS_ON_TOP: WindowMode = WindowMode {
        borderless: true,
        transparent: false,
        always_on_top: true,
    };

    fn laptop() -> MonitorConfig {
        MonitorConfig::new(vec![monitor("Monitor #41003", 0, 0, 1728, 1117)])
    }

    fn docked() -> MonitorConfig {
        MonitorConfig::new(vec![
            monitor("Monitor #41003", 0, 0, 1728, 1117),
            monitor("Monitor #16642", 1728, -200, 2560, 1440),
        ])
    }

    // ---- config key ---------------------------------------------------------

    #[test]
    fn key_is_readable_and_independent_of_enumeration_order() {
        let a = monitor("Monitor #41003", 0, 0, 1728, 1117);
        let b = monitor("Monitor #16642", 1728, -200, 2560, 1440);
        let forward = MonitorConfig::new(vec![a.clone(), b.clone()]);
        let reverse = MonitorConfig::new(vec![b, a]);
        assert_eq!(forward.key(), reverse.key());
        assert_eq!(
            forward.key(),
            "Monitor #41003 1728x1117@0,0 + Monitor #16642 2560x1440@1728,-200"
        );
    }

    #[test]
    fn identical_models_at_different_positions_stay_distinct() {
        let left = monitor("Monitor #16642", 0, 0, 2560, 1440);
        let right = monitor("Monitor #16642", 2560, 0, 2560, 1440);
        let both = MonitorConfig::new(vec![left.clone(), right.clone()]);
        assert_eq!(
            both.key(),
            "Monitor #16642 2560x1440@0,0 + Monitor #16642 2560x1440@2560,0"
        );
        // Either one alone is its own configuration, and they differ from each other.
        let only_left = MonitorConfig::new(vec![left]);
        let only_right = MonitorConfig::new(vec![right]);
        assert_ne!(only_left.key(), only_right.key());
        assert_ne!(only_left.key(), both.key());
    }

    #[test]
    fn scale_factor_is_not_part_of_the_key() {
        let mut retina = monitor("Monitor #41003", 0, 0, 1728, 1117);
        let plain = MonitorConfig::new(vec![retina.clone()]);
        retina.physical_px_per_unit = 2.0;
        assert_eq!(MonitorConfig::new(vec![retina]).key(), plain.key());
    }

    // ---- visibility fit -----------------------------------------------------

    #[test]
    fn fit_keeps_visible_and_straddling_windows_in_place() {
        let config = docked();
        let on_laptop = geometry(100, 80, 640, 360);
        assert_eq!(config.fit(on_laptop), on_laptop);
        // Straddling the laptop/external boundary is a legitimate placement.
        let straddling = geometry(1500, 100, 640, 360);
        assert_eq!(config.fit(straddling), straddling);
    }

    #[test]
    fn fit_recenters_off_screen_window_on_primary() {
        // Saved on the external display, which is no longer connected.
        let fitted = laptop().fit(geometry(2500, 100, 640, 360));
        assert_eq!(
            fitted,
            geometry((1728 - 640) / 2, (1117 - 360) / 2, 640, 360)
        );
    }

    #[test]
    fn fit_needs_more_than_a_sliver_on_screen() {
        // Only 10 units overlap the laptop horizontally: not enough to grab.
        let fitted = laptop().fit(geometry(1718, 100, 640, 360));
        assert_eq!(fitted.x, (1728 - 640) / 2);
        // 64 units visible is enough.
        let grabbable = geometry(1728 - 64, 100, 640, 360);
        assert_eq!(laptop().fit(grabbable), grabbable);
    }

    #[test]
    fn fit_caps_size_to_display_and_floors_zero() {
        // A stale 4K-sized rect on the laptop panel is capped to the panel.
        let fitted = laptop().fit(geometry(0, 0, 3840, 2160));
        assert_eq!(fitted, geometry(0, 0, 1728, 1117));
        // Zero size floors to 1×1, even with no displays known.
        assert_eq!(
            MonitorConfig::default().fit(geometry(5, 5, 0, 0)),
            geometry(5, 5, 1, 1)
        );
    }

    #[test]
    fn fit_recenters_when_size_cap_pulls_window_off_screen() {
        // Huge window whose only visible part is its right end; capping the width
        // keeps the left edge, which would leave nothing visible.
        let fitted = laptop().fit(geometry(-3000, 0, 4000, 500));
        assert_eq!(fitted, geometry(0, (1117 - 500) / 2, 1728, 500));
    }

    // ---- window_state.json formats + migration -----------------------------

    fn parse(json: &str) -> Option<StoredWindowState> {
        serde_json::from_str(json).ok()
    }

    #[test]
    fn current_format_roundtrips_without_touching_legacy_flags() {
        let mut profiles = WindowProfiles::default();
        profiles.insert_profile(
            laptop().key().to_owned(),
            WindowPlacement {
                geometry: Some(geometry(200, 120, 640, 360)),
                mode: BORDERLESS_ON_TOP,
            },
        );
        profiles.insert_profile(
            "mode-only".to_owned(),
            WindowPlacement {
                geometry: None,
                mode: WindowMode::default(),
            },
        );
        profiles.last_config = Some(laptop().key().to_owned());
        let json = serde_json::to_string(&profiles).unwrap();
        // Flat, human-readable profile entries.
        assert!(json.contains(r#""x":200,"y":120,"width":640,"height":360,"borderless":true"#));

        let back = WindowProfiles::from_stored(parse(&json), || {
            panic!("legacy app_state flags must not be read for the current format")
        });
        assert_eq!(back, profiles);
    }

    #[test]
    fn legacy_geometry_and_app_state_flags_seed_the_launch_profile() {
        let legacy = r#"{"x":400,"y":200,"width":2560,"height":1440}"#;
        let mut profiles = WindowProfiles::from_stored(parse(legacy), || BORDERLESS_ON_TOP);
        // Physical pixels saved on a 2× primary display → points.
        let retina = MonitorConfig::new(vec![MonitorDescriptor {
            physical_px_per_unit: 2.0,
            ..monitor("Monitor #41003", 0, 0, 1728, 1117)
        }]);
        assert!(profiles.adopt_legacy(&retina));
        assert_eq!(
            profiles.launch_placement(retina.key()),
            WindowPlacement {
                geometry: Some(geometry(200, 100, 1280, 720)),
                mode: BORDERLESS_ON_TOP,
            }
        );
        assert_eq!(profiles.last_config.as_deref(), Some(retina.key()));
        // Adoption happens once.
        assert!(!profiles.adopt_legacy(&docked()));
    }

    #[test]
    fn missing_window_state_still_carries_old_chrome_flags() {
        let mut profiles = WindowProfiles::from_stored(None, || BORDERLESS_ON_TOP);
        assert!(profiles.adopt_legacy(&laptop()));
        assert_eq!(
            profiles.launch_placement(laptop().key()),
            WindowPlacement {
                geometry: None,
                mode: BORDERLESS_ON_TOP,
            }
        );
    }

    #[test]
    fn fresh_install_has_nothing_to_adopt() {
        let mut profiles = WindowProfiles::from_stored(None, WindowMode::default);
        assert!(!profiles.adopt_legacy(&laptop()));
        assert_eq!(profiles, WindowProfiles::default());
        // An empty object (or one with unknown fields) is the current format.
        let empty = WindowProfiles::from_stored(parse(r#"{"future":1}"#), || unreachable!());
        assert_eq!(empty, WindowProfiles::default());
    }

    #[test]
    fn profile_with_missing_flags_or_partial_geometry_defaults() {
        let json = r#"{"profiles":{"k":{"x":1,"y":2,"width":3},"m":{"always_on_top":true}}}"#;
        let profiles = WindowProfiles::from_stored(parse(json), || unreachable!());
        assert_eq!(profiles.profiles["k"].placement, WindowPlacement::default());
        assert_eq!(
            profiles.profiles["m"].placement,
            WindowPlacement {
                geometry: None,
                mode: WindowMode {
                    always_on_top: true,
                    ..WindowMode::default()
                },
            }
        );
    }

    #[test]
    fn launch_falls_back_to_last_used_chrome_without_geometry() {
        let mut profiles = WindowProfiles::default();
        profiles.insert_profile(
            docked().key().to_owned(),
            WindowPlacement {
                geometry: Some(geometry(2000, 0, 640, 360)),
                mode: BORDERLESS_ON_TOP,
            },
        );
        profiles.last_config = Some(docked().key().to_owned());
        assert_eq!(
            profiles.launch_placement(laptop().key()),
            WindowPlacement {
                geometry: None,
                mode: BORDERLESS_ON_TOP,
            }
        );
        // No last config at all → defaults.
        profiles.last_config = None;
        assert_eq!(
            profiles.launch_placement(laptop().key()),
            WindowPlacement::default()
        );
    }

    // ---- tracker ------------------------------------------------------------

    /// A tracker whose saves are captured for inspection.
    fn tracker(profiles: WindowProfiles) -> (PlacementTracker, Rc<RefCell<Vec<WindowProfiles>>>) {
        let saves = Rc::new(RefCell::new(Vec::new()));
        let sink = saves.clone();
        let tracker = PlacementTracker::new(profiles, move |p| sink.borrow_mut().push(p.clone()));
        (tracker, saves)
    }

    fn snapshot(monitors: MonitorConfig, g: WindowGeometry) -> impl FnOnce() -> WindowSnapshot {
        move || WindowSnapshot {
            monitors,
            geometry: Some(g),
        }
    }

    fn saved_geometry(saves: &RefCell<Vec<WindowProfiles>>, key: &str) -> Option<WindowGeometry> {
        saves.borrow().last()?.profiles.get(key)?.placement.geometry
    }

    #[test]
    fn start_restores_fitted_profile() {
        let mut profiles = WindowProfiles::default();
        profiles.insert_profile(
            laptop().key().to_owned(),
            WindowPlacement {
                geometry: Some(geometry(0, 0, 4000, 4000)),
                mode: BORDERLESS_ON_TOP,
            },
        );
        let (mut t, saves) = tracker(profiles);
        let placement = t.start(laptop(), Instant::now());
        assert_eq!(placement.geometry, Some(geometry(0, 0, 1728, 1117)));
        assert_eq!(placement.mode, BORDERLESS_ON_TOP);
        // Nothing migrated → no write at launch.
        assert!(saves.borrow().is_empty());
    }

    #[test]
    fn same_config_commits_geometry_after_a_confirming_check() {
        let (mut t, saves) = tracker(WindowProfiles::default());
        let t0 = Instant::now();
        t.start(laptop(), t0);
        let g1 = geometry(10, 20, 640, 360);
        // First poll checks immediately; its reading is only a candidate.
        assert_eq!(t.poll(t0, BORDERLESS_ON_TOP, snapshot(laptop(), g1)), None);
        assert_eq!(saved_geometry(&saves, laptop().key()), None);
        // Before the next check is due, the window is not even queried.
        let not_due = t0 + MONITOR_CHECK_INTERVAL / 2;
        t.poll(not_due, BORDERLESS_ON_TOP, || panic!("check not due"));
        // The next check on the same config confirms (commits) the previous reading
        // and starts the launch config's profile.
        let g2 = geometry(30, 40, 640, 360);
        let t1 = t0 + MONITOR_CHECK_INTERVAL;
        t.poll(t1, BORDERLESS_ON_TOP, snapshot(laptop(), g2));
        assert_eq!(saved_geometry(&saves, laptop().key()), Some(g1));
        assert_eq!(
            saves.borrow().last().unwrap().profiles[laptop().key()]
                .placement
                .mode,
            BORDERLESS_ON_TOP
        );
        let t2 = t1 + MONITOR_CHECK_INTERVAL;
        t.poll(t2, BORDERLESS_ON_TOP, snapshot(laptop(), g2));
        assert_eq!(saved_geometry(&saves, laptop().key()), Some(g2));
    }

    #[test]
    fn os_induced_geometry_after_display_loss_never_reaches_old_profile() {
        let (mut t, saves) = tracker(WindowProfiles::default());
        let t0 = Instant::now();
        let mode = WindowMode::default();
        t.start(docked(), t0);
        // Last good placement: on the external display, confirmed by two checks.
        let on_external = geometry(2000, 100, 1280, 720);
        t.poll(t0, mode, snapshot(docked(), on_external));
        let t1 = t0 + MONITOR_CHECK_INTERVAL;
        t.poll(t1, mode, snapshot(docked(), on_external));
        assert_eq!(saved_geometry(&saves, docked().key()), Some(on_external));

        // External unplugged; the OS shoved the window onto the laptop.
        let shoved = geometry(0, 25, 1280, 720);
        let t2 = t1 + MONITOR_CHECK_INTERVAL;
        assert_eq!(t.poll(t2, mode, snapshot(laptop(), shoved)), None);
        // Settled: switch, with nothing committed to the old profile.
        let t3 = t2 + MONITOR_CONFIG_SETTLE;
        assert_eq!(t.poll(t3, mode, snapshot(laptop(), shoved)), None);
        let last = saves.borrow().last().unwrap().clone();
        // The old profile kept its last good geometry; the new config got a profile
        // from the window as the OS left it.
        assert_eq!(
            last.profiles[docked().key()].placement.geometry,
            Some(on_external)
        );
        assert_eq!(
            last.profiles[laptop().key()].placement.geometry,
            Some(shoved)
        );
        assert_eq!(last.last_config.as_deref(), Some(laptop().key()));
    }

    #[test]
    fn os_move_seen_before_the_display_list_updates_is_never_committed() {
        // The OS moved the window (and sent ScaleFactorChanged) while the display list
        // still read as the old config; the next check sees the new config.
        let (mut t, saves) = tracker(WindowProfiles::default());
        let t0 = Instant::now();
        let mode = WindowMode::default();
        t.start(docked(), t0);
        let on_external = geometry(2000, 100, 1280, 720);
        t.poll(t0, mode, snapshot(docked(), on_external));
        let t1 = t0 + MONITOR_CHECK_INTERVAL;
        t.poll(t1, mode, snapshot(docked(), on_external));

        let shoved = geometry(0, 25, 1280, 720);
        t.request_check();
        let t2 = t1 + Duration::from_millis(10);
        t.poll(t2, mode, snapshot(docked(), shoved));
        let t3 = t2 + MONITOR_CHECK_INTERVAL;
        t.poll(t3, mode, snapshot(laptop(), shoved));
        let t4 = t3 + MONITOR_CONFIG_SETTLE;
        t.poll(t4, mode, snapshot(laptop(), shoved));
        assert_eq!(saved_geometry(&saves, docked().key()), Some(on_external));
        assert_eq!(t.profiles.last_config.as_deref(), Some(laptop().key()));
    }

    #[test]
    fn display_blip_restores_saved_placement_instead_of_committing_moved_rect() {
        let (mut t, saves) = tracker(WindowProfiles::default());
        let t0 = Instant::now();
        let mode = BORDERLESS_ON_TOP;
        t.start(docked(), t0);
        let on_external = geometry(2000, 100, 1280, 720);
        t.poll(t0, mode, snapshot(docked(), on_external));
        let t1 = t0 + MONITOR_CHECK_INTERVAL;
        t.poll(t1, mode, snapshot(docked(), on_external));

        // The external display drops out for one check; the OS moves the window.
        let shoved = geometry(0, 25, 1280, 720);
        let t2 = t1 + MONITOR_CHECK_INTERVAL;
        assert_eq!(t.poll(t2, mode, snapshot(laptop(), shoved)), None);
        // Back before settling, but the window is still where the OS put it: move it
        // back to the saved placement rather than commit the moved rect.
        let t3 = t2 + MONITOR_CHECK_INTERVAL / 2;
        t.request_check();
        assert_eq!(
            t.poll(t3, mode, snapshot(docked(), shoved)),
            Some(WindowPlacement {
                geometry: Some(on_external),
                mode,
            })
        );
        // Two more checks with the window back in place: the moved rect never shows
        // up, and no profile was started for the blip's config.
        let t4 = t3 + MONITOR_CHECK_INTERVAL;
        assert_eq!(t.poll(t4, mode, snapshot(docked(), on_external)), None);
        let t5 = t4 + MONITOR_CHECK_INTERVAL;
        assert_eq!(t.poll(t5, mode, snapshot(docked(), on_external)), None);
        for saved in saves.borrow().iter() {
            assert_ne!(
                saved
                    .profiles
                    .get(docked().key())
                    .and_then(|p| p.placement.geometry),
                Some(shoved)
            );
            assert!(!saved.profiles.contains_key(laptop().key()));
        }
        assert_eq!(
            t.profiles.profiles[docked().key()].placement.geometry,
            Some(on_external)
        );
    }

    #[test]
    fn returning_to_a_known_config_applies_its_profile() {
        let mut profiles = WindowProfiles::default();
        let docked_placement = WindowPlacement {
            geometry: Some(geometry(2000, 100, 1280, 720)),
            mode: BORDERLESS_ON_TOP,
        };
        profiles.insert_profile(docked().key().to_owned(), docked_placement);
        let (mut t, _saves) = tracker(profiles);
        let t0 = Instant::now();
        t.start(laptop(), t0);
        let on_laptop = geometry(100, 100, 640, 360);
        t.poll(t0, WindowMode::default(), snapshot(laptop(), on_laptop));
        let confirmed = t0 + MONITOR_CHECK_INTERVAL;
        t.poll(
            confirmed,
            WindowMode::default(),
            snapshot(laptop(), on_laptop),
        );

        // External plugged back in; a scale change hint asks for a check right away.
        t.request_check();
        let t1 = confirmed + Duration::from_millis(10);
        assert_eq!(
            t.poll(t1, WindowMode::default(), snapshot(docked(), on_laptop)),
            None
        );
        let t2 = t1 + MONITOR_CONFIG_SETTLE;
        assert_eq!(
            t.poll(t2, WindowMode::default(), snapshot(docked(), on_laptop)),
            Some(docked_placement)
        );
        let mode_of = |t: &PlacementTracker, config: MonitorConfig| {
            t.profiles.profiles[config.key()].placement.mode
        };
        // Chrome toggles now belong to the docked profile only.
        t.record_mode(WindowMode::default(), t2);
        assert_eq!(mode_of(&t, docked()), WindowMode::default());
        assert_eq!(mode_of(&t, laptop()), WindowMode::default());
        t.record_mode(BORDERLESS_ON_TOP, t2);
        assert_eq!(mode_of(&t, docked()), BORDERLESS_ON_TOP);
        assert_eq!(mode_of(&t, laptop()), WindowMode::default());
    }

    #[test]
    fn chrome_toggle_while_switch_pending_is_kept_in_the_new_profile() {
        let mut profiles = WindowProfiles::default();
        let docked_geometry = geometry(2000, 100, 1280, 720);
        profiles.insert_profile(
            docked().key().to_owned(),
            WindowPlacement {
                geometry: Some(docked_geometry),
                mode: WindowMode::default(),
            },
        );
        let (mut t, saves) = tracker(profiles);
        let t0 = Instant::now();
        t.start(laptop(), t0);
        let on_laptop = geometry(100, 100, 640, 360);
        t.poll(t0, WindowMode::default(), snapshot(laptop(), on_laptop));

        // Docked is seen, and the user toggles chrome before the switch settles.
        let t1 = t0 + MONITOR_CHECK_INTERVAL;
        t.poll(t1, WindowMode::default(), snapshot(docked(), on_laptop));
        t.record_mode(BORDERLESS_ON_TOP, t1 + Duration::from_millis(100));
        // The switch restores docked's geometry but keeps the toggled chrome, and
        // writes it into docked's profile.
        let t2 = t1 + MONITOR_CONFIG_SETTLE;
        assert_eq!(
            t.poll(t2, BORDERLESS_ON_TOP, snapshot(docked(), on_laptop)),
            Some(WindowPlacement {
                geometry: Some(docked_geometry),
                mode: BORDERLESS_ON_TOP,
            })
        );
        let last = saves.borrow().last().unwrap().clone();
        assert_eq!(
            last.profiles[docked().key()].placement.mode,
            BORDERLESS_ON_TOP
        );
        // The flag is spent: a later switch applies saved chrome again.
        assert!(!t.mode_changed_while_settling);
    }

    #[test]
    fn transient_config_reading_does_not_switch_or_commit() {
        let (mut t, saves) = tracker(WindowProfiles::default());
        let t0 = Instant::now();
        let mode = WindowMode::default();
        t.start(laptop(), t0);
        let good = geometry(10, 10, 640, 360);
        t.poll(t0, mode, snapshot(laptop(), good));
        let t1 = t0 + MONITOR_CHECK_INTERVAL;
        t.poll(t1, mode, snapshot(laptop(), good));
        let saves_before = saves.borrow().len();

        // One odd reading (display briefly reported at another size)…
        let blip = MonitorConfig::new(vec![monitor("Monitor #41003", 0, 0, 3456, 2234)]);
        let t2 = t1 + MONITOR_CHECK_INTERVAL;
        t.poll(t2, mode, snapshot(blip, geometry(0, 0, 9, 9)));
        // …and an empty enumeration mid-reconfiguration.
        let t3 = t2 + MONITOR_CHECK_INTERVAL;
        t.poll(t3, mode, || WindowSnapshot {
            monitors: MonitorConfig::default(),
            geometry: Some(geometry(0, 0, 9, 9)),
        });
        // Back to normal: no switch, the saved placement is re-applied, and nothing
        // was written meanwhile.
        let t4 = t3 + MONITOR_CHECK_INTERVAL;
        assert_eq!(
            t.poll(t4, mode, snapshot(laptop(), good)),
            Some(WindowPlacement {
                geometry: Some(good),
                mode,
            })
        );
        assert_eq!(saves.borrow().len(), saves_before);
        assert_eq!(t.profiles.profiles.len(), 1);
    }

    #[test]
    fn finish_commits_only_for_the_current_config() {
        let (mut t, saves) = tracker(WindowProfiles::default());
        let t0 = Instant::now();
        let mode = WindowMode::default();
        t.start(docked(), t0);
        let good = geometry(2000, 100, 640, 360);
        t.poll(t0, mode, snapshot(docked(), good));
        let t1 = t0 + MONITOR_CHECK_INTERVAL;
        t.poll(t1, mode, snapshot(docked(), good));
        // Quit right after the OS moved the window off an unplugged display.
        t.finish(
            t1 + Duration::from_millis(1),
            mode,
            WindowSnapshot {
                monitors: laptop(),
                geometry: Some(geometry(0, 0, 640, 360)),
            },
        );
        assert_eq!(saved_geometry(&saves, docked().key()), Some(good));
        // Quit on the current config saves the latest geometry right away, without a
        // confirming check and despite the debounce.
        let moved = geometry(2100, 100, 640, 360);
        t.finish(
            t1 + Duration::from_millis(2),
            mode,
            WindowSnapshot {
                monitors: docked(),
                geometry: Some(moved),
            },
        );
        assert_eq!(saved_geometry(&saves, docked().key()), Some(moved));
    }

    #[test]
    fn finish_during_a_display_blip_does_not_commit() {
        let (mut t, saves) = tracker(WindowProfiles::default());
        let t0 = Instant::now();
        let mode = WindowMode::default();
        t.start(docked(), t0);
        let good = geometry(2000, 100, 640, 360);
        t.poll(t0, mode, snapshot(docked(), good));
        let t1 = t0 + MONITOR_CHECK_INTERVAL;
        t.poll(t1, mode, snapshot(docked(), good));
        let t2 = t1 + MONITOR_CHECK_INTERVAL;
        t.poll(t2, mode, snapshot(laptop(), geometry(0, 0, 640, 360)));
        // The displays read as docked again at quit, but the window may still be
        // where the OS put it during the blip.
        t.finish(
            t2 + Duration::from_millis(1),
            mode,
            WindowSnapshot {
                monitors: docked(),
                geometry: Some(geometry(0, 0, 640, 360)),
            },
        );
        assert_eq!(saved_geometry(&saves, docked().key()), Some(good));
    }

    #[test]
    fn startup_migration_is_saved_immediately() {
        let profiles = WindowProfiles::from_stored(
            parse(r#"{"x":10,"y":20,"width":640,"height":360}"#),
            || BORDERLESS_ON_TOP,
        );
        let (mut t, saves) = tracker(profiles);
        let placement = t.start(laptop(), Instant::now());
        assert_eq!(placement.geometry, Some(geometry(10, 20, 640, 360)));
        assert_eq!(placement.mode, BORDERLESS_ON_TOP);
        assert_eq!(
            saved_geometry(&saves, laptop().key()),
            Some(placement.geometry.unwrap())
        );
    }

    #[test]
    fn launch_without_displays_defers_config_and_legacy_adoption() {
        let profiles = WindowProfiles::from_stored(
            parse(r#"{"x":400,"y":200,"width":2560,"height":1440}"#),
            || BORDERLESS_ON_TOP,
        );
        let (mut t, saves) = tracker(profiles);
        let t0 = Instant::now();
        // Neither the display list nor the primary monitor was available.
        let placement = t.start(MonitorConfig::default(), t0);
        assert_eq!(
            placement,
            WindowPlacement {
                geometry: None,
                mode: BORDERLESS_ON_TOP,
            }
        );
        // Not migrated (no scale to convert with), nothing filed under the empty key.
        assert!(t.profiles.legacy.is_some());
        t.poll(t0, BORDERLESS_ON_TOP, || WindowSnapshot {
            monitors: MonitorConfig::default(),
            geometry: Some(geometry(5, 5, 640, 360)),
        });
        t.record_mode(WindowMode::default(), t0);
        t.finish(
            t0,
            WindowMode::default(),
            WindowSnapshot {
                monitors: MonitorConfig::default(),
                geometry: Some(geometry(5, 5, 640, 360)),
            },
        );
        assert!(t.profiles.profiles.is_empty());
        assert!(saves.borrow().is_empty());

        // Displays show up (a 2× panel): switch to them as on a hotplug and adopt the
        // legacy rect with their scale; the chrome toggled meanwhile is kept.
        let retina = MonitorConfig::new(vec![MonitorDescriptor {
            physical_px_per_unit: 2.0,
            ..monitor("Monitor #41003", 0, 0, 1728, 1117)
        }]);
        let at_default = geometry(224, 198, 1280, 720);
        let t1 = t0 + MONITOR_CHECK_INTERVAL;
        t.poll(
            t1,
            WindowMode::default(),
            snapshot(retina.clone(), at_default),
        );
        let t2 = t1 + MONITOR_CONFIG_SETTLE;
        assert_eq!(
            t.poll(
                t2,
                WindowMode::default(),
                snapshot(retina.clone(), at_default)
            ),
            Some(WindowPlacement {
                geometry: Some(geometry(200, 100, 1280, 720)),
                mode: WindowMode::default(),
            })
        );
        let last = saves.borrow().last().unwrap().clone();
        assert_eq!(last.last_config.as_deref(), Some(retina.key()));
        assert_eq!(last.profiles.keys().collect::<Vec<_>>(), vec![retina.key()]);
        assert!(t.profiles.legacy.is_none());
    }

    #[test]
    fn prune_drops_least_recently_used_but_never_the_current_profile() {
        let mut profiles = WindowProfiles::default();
        for i in 0..MAX_PROFILES + 4 {
            profiles.insert_profile(format!("k{i:02}"), WindowPlacement::default());
        }
        // The current config is the oldest one.
        profiles.last_config = Some("k00".to_owned());
        profiles.prune();
        let kept: Vec<_> = profiles.profiles.keys().cloned().collect();
        assert_eq!(kept.len(), MAX_PROFILES);
        assert!(kept.contains(&"k00".to_owned()));
        for dropped in ["k01", "k02", "k03", "k04"] {
            assert!(!kept.contains(&dropped.to_owned()), "{dropped} kept");
        }
    }

    #[test]
    fn old_file_without_use_counts_is_pruned_on_save_keeping_current() {
        // Written before pruning: no `use_counter`, no per-profile `last_used`.
        let mut entries = Vec::new();
        for i in 0..MAX_PROFILES + 4 {
            entries.push(format!(r#""k{i:02}":{{"x":1,"y":2,"width":3,"height":4}}"#));
        }
        entries.push(format!(r#""{}":{{"borderless":true}}"#, laptop().key()));
        let json = format!(r#"{{"profiles":{{{}}}}}"#, entries.join(","));
        let profiles = WindowProfiles::from_stored(parse(&json), || unreachable!());
        assert_eq!(profiles.use_counter, 0);
        assert!(profiles.profiles.values().all(|p| p.last_used == 0));
        assert_eq!(
            profiles.profiles["k00"].placement.geometry,
            Some(geometry(1, 2, 3, 4))
        );

        let (mut t, saves) = tracker(profiles);
        let t0 = Instant::now();
        t.start(laptop(), t0);
        // The launch is recorded as a use and flushed by the first poll.
        t.poll(t0, WindowMode::default(), || WindowSnapshot {
            monitors: laptop(),
            geometry: None,
        });
        let last = saves.borrow().last().unwrap().clone();
        assert_eq!(last.profiles.len(), MAX_PROFILES);
        assert!(last.profiles[laptop().key()].last_used > 0);
        assert!(last.profiles[laptop().key()].placement.mode.borderless);
        let json = serde_json::to_string(&last).unwrap();
        assert!(json.contains(r#""use_counter":1"#));
        assert!(json.contains(r#""last_used":1"#));
    }

    #[test]
    fn untracked_tracker_is_inert() {
        // Before start (e.g. VizApp in tests) nothing is queried or recorded.
        let mut t = PlacementTracker::default();
        let now = Instant::now();
        assert_eq!(
            t.poll(now, WindowMode::default(), || panic!("not started")),
            None
        );
        t.record_mode(BORDERLESS_ON_TOP, now);
        assert!(t.profiles.profiles.is_empty());
    }
}
