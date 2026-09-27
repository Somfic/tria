//! Screenshot-on-a-timer, for checking what a scene looks like without a
//! person at the window.
//!
//! Driven entirely by environment variables under a per-app prefix, so a
//! normal run never touches any of it:
//!
//! - `<PREFIX>_SCREENSHOT=<path>` enables the harness
//! - `<PREFIX>_SCREENSHOT_FRAME=<n>` which frame to capture (default 90)
//!
//! An app stages its own scene by adding systems `.after(Harness)` and
//! reading [`Shot::frame`], [`Shot::parse`] and [`Shot::flag`] -- which keep
//! the prefix in one place, so `shot.parse::<f64>("WARP")` reads
//! `<PREFIX>_WARP` without the app repeating it.
//!
//! Nothing is inserted when the harness is off, so app staging should take
//! `Option<Res<Shot>>` and do nothing when it is absent. That is the right
//! behaviour anyway: these variables should only bite a capture run.

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use std::str::FromStr;

const DEFAULT_FRAME: u32 = 90;

/// Frames to leave the app running after the capture is issued. The save is
/// asynchronous, so exiting immediately truncates the file.
const DRAIN_FRAMES: u32 = 30;

/// The harness's own bookkeeping: the frame counter, the capture and the
/// exit, as one system. App staging runs `.after` this so it sees the same
/// frame number the capture will use.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Harness;

pub struct ShotPlugin {
    prefix: String,
}

impl ShotPlugin {
    /// `prefix` is the environment variable namespace, e.g. `"SSA"`.
    pub fn new(prefix: &str) -> Self {
        ShotPlugin {
            prefix: prefix.to_owned(),
        }
    }
}

impl Plugin for ShotPlugin {
    fn build(&self, app: &mut App) {
        let prefix = self.prefix.clone();
        let Ok(path) = std::env::var(format!("{prefix}_SCREENSHOT")) else {
            return;
        };
        let at = std::env::var(format!("{prefix}_SCREENSHOT_FRAME"))
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_FRAME);

        app.insert_resource(Shot {
            prefix,
            path,
            at,
            frame: 0,
            exit_at: None,
        })
        .add_systems(Update, run.in_set(Harness));
    }
}

#[derive(Resource)]
pub struct Shot {
    prefix: String,
    path: String,
    at: u32,
    frame: u32,
    exit_at: Option<u32>,
}

impl Shot {
    /// Frames rendered so far. 1 on the first.
    pub fn frame(&self) -> u32 {
        self.frame
    }

    /// The frame the capture happens on.
    pub fn at(&self) -> u32 {
        self.at
    }

    /// True on exactly the captured frame, for logging state that explains a
    /// shot.
    pub fn capturing(&self) -> bool {
        self.frame == self.at
    }

    /// One of this harness's variables, without the prefix.
    pub fn var(&self, name: &str) -> Option<String> {
        std::env::var(format!("{}_{name}", self.prefix)).ok()
    }

    /// One of this harness's variables, parsed. `None` if unset or unparseable.
    pub fn parse<T: FromStr>(&self, name: &str) -> Option<T> {
        self.var(name).and_then(|v| v.parse().ok())
    }

    /// Whether one of this harness's variables is set at all.
    pub fn flag(&self, name: &str) -> bool {
        self.var(name).is_some()
    }
}

fn run(mut commands: Commands, mut shot: ResMut<Shot>, mut exit: MessageWriter<AppExit>) {
    shot.frame += 1;

    if shot.frame == shot.at {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(shot.path.clone()));
        shot.exit_at = Some(shot.frame + DRAIN_FRAMES);
    }

    // A run driven by these variables is never interactive, so leaving it to
    // a person to close the window is just a way to forget. Compared with
    // `>=`, not `==`: miss the one frame that matches and it never exits.
    if shot.exit_at.is_some_and(|at| shot.frame >= at) {
        exit.write(AppExit::Success);
    }
}
