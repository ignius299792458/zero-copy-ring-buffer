// One RL transition.
// repr(C) pins the field order and alignment so Rust, Python, and PyTorch
// all agree on where every byte lives — critical for zero-copy reads later.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PpoTransition {
    pub observation: [f32; 4], // CartPole: x, x_dot, theta, theta_dot
    pub action: f32,           // 0.0 = left, 1.0 = right
    pub log_prob: f32,         // ln π_old(a|s) — frozen at collection time
    pub reward: f32,           // +1.0 per surviving step
    pub done: f32,             // 1.0 = episode ended, 0.0 = still running
}

impl PpoTransition {
    pub fn zero() -> Self {
        Self {
            observation: [0.0; 4],
            action: 0.0,
            log_prob: 0.0,
            reward: 0.0,
            done: 0.0,
        }
    }
}
