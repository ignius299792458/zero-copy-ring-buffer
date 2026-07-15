// One RL transition — fixed-size in memory, flexible in observation length.
// observation is inline [f32; MAX_OBS_DIM] (no heap, no pointer) so the
// struct stays repr(C)-stable and Copy. obs_dim records how many entries
// are actually used; unused tail entries are zero.

pub const MAX_OBS_DIM: usize = 20;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StateTransition {
    pub observation: [f32; MAX_OBS_DIM],
    pub obs_dim: u32,
    pub action: f32,
    pub log_prob: f32,
    pub reward: f32,
    pub done: f32,
}

impl StateTransition {
    pub fn zero() -> Self {
        Self {
            observation: [0.0; MAX_OBS_DIM],
            obs_dim: 0,
            action: 0.0,
            log_prob: 0.0,
            reward: 0.0,
            done: 0.0,
        }
    }
}
