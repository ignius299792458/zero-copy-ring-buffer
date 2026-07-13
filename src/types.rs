// One RL transition.
// repr(C) pins the field order and alignment so Rust, Python, and PyTorch
// all agree on where every byte lives — critical for zero-copy reads later.
use std::collections::HashMap;

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct StateTransitionDTO<O: Default> {
    pub observation: O, // caller decides: Vec<f32>, ndarray, bytes, whatever
    pub metadata: HashMap<String, f32>, // log_prob, reward, done(0.0/1.0), extensible
}
