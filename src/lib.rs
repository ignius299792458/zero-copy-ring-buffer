mod ring_buffer;
mod types;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ring_buffer::RingBuffer;
use types::{StateTransition, MAX_OBS_DIM};

/// Python-visible wrapper for one StateTransition.
/// observation is a variable-length list (len <= MAX_OBS_DIM); the tail
/// is padded with zeros internally, obs_dim records the real length.
#[pyclass(name = "StateTransition")]
#[derive(Clone)]
struct PyTransition {
    inner: StateTransition,
}

#[pymethods]
impl PyTransition {
    #[new]
    fn new(
        observation: Vec<f32>,
        action: f32,
        log_prob: f32,
        reward: f32,
        done: f32,
    ) -> PyResult<Self> {
        let d = observation.len();
        if d > MAX_OBS_DIM {
            return Err(PyValueError::new_err(format!(
                "observation length {d} exceeds MAX_OBS_DIM {MAX_OBS_DIM}"
            )));
        }
        let mut obs = [0.0f32; MAX_OBS_DIM];
        obs[..d].copy_from_slice(&observation);

        Ok(Self {
            inner: StateTransition {
                observation: obs,
                obs_dim: d as u32,
                action,
                log_prob,
                reward,
                done,
            },
        })
    }

    /// Returns only the used portion of the observation.
    #[getter]
    fn observation(&self) -> Vec<f32> {
        self.inner.observation[..self.inner.obs_dim as usize].to_vec()
    }
    #[getter]
    fn obs_dim(&self) -> u32 {
        self.inner.obs_dim
    }
    #[getter]
    fn action(&self) -> f32 {
        self.inner.action
    }
    #[getter]
    fn log_prob(&self) -> f32 {
        self.inner.log_prob
    }
    #[getter]
    fn reward(&self) -> f32 {
        self.inner.reward
    }
    #[getter]
    fn done(&self) -> f32 {
        self.inner.done
    }

    fn __repr__(&self) -> String {
        format!(
            "Transition(obs_dim={}, obs={:?}, a={}, lp={:.3}, r={}, done={})",
            self.inner.obs_dim,
            &self.inner.observation[..self.inner.obs_dim as usize],
            self.inner.action,
            self.inner.log_prob,
            self.inner.reward,
            self.inner.done
        )
    }
}

/// Python-visible wrapper for the RingBuffer.
/// push rejects writes when full (returns False); pop returns None when empty.
#[pyclass(name = "RingBuffer")]
struct PyRingBuffer {
    buf: RingBuffer,
}

#[pymethods]
impl PyRingBuffer {
    #[new]
    fn new(capacity: usize) -> Self {
        Self {
            buf: RingBuffer::new(capacity),
        }
    }

    /// Push one transition. Returns False if the buffer is full (write rejected).
    fn push(&mut self, t: &PyTransition) -> bool {
        self.buf.push(t.inner).is_ok()
    }

    /// Pop the oldest transition, or None if the buffer is empty.
    fn pop(&mut self) -> Option<PyTransition> {
        self.buf.pop().map(|inner| PyTransition { inner })
    }

    /// Pop up to n transitions at once.
    fn pop_batch(&mut self, n: usize) -> Vec<PyTransition> {
        (0..n)
            .filter_map(|_| self.buf.pop())
            .map(|inner| PyTransition { inner })
            .collect()
    }

    #[getter]
    fn len(&self) -> usize {
        self.buf.len()
    }
    #[getter]
    fn capacity(&self) -> usize {
        self.buf.get_capacity()
    }
    #[getter]
    fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
    #[getter]
    fn is_full(&self) -> bool {
        self.buf.is_full()
    }

    fn __repr__(&self) -> String {
        format!(
            "RingBuffer(len={}, capacity={}, full={})",
            self.buf.len(),
            self.buf.get_capacity(),
            self.buf.is_full()
        )
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTransition>()?;
    m.add_class::<PyRingBuffer>()?;
    m.add("MAX_OBS_DIM", MAX_OBS_DIM)?;
    Ok(())
}
