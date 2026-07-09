// PyO3 bridge.
// module name MUST be "_core" to match pyproject.toml:
//   module-name = "zero_copy_buffer._core"
// Maturin installs this as zero_copy_buffer/_core.so
// python/zero_copy_buffer/__init__.py re-exports from here.

mod ring_buffer;
mod types;

use pyo3::prelude::*;
use ring_buffer::SimpleBuffer;
use types::PpoTransition;

/// Python-visible wrapper for one PpoTransition.
#[pyclass(name = "Transition")]
#[derive(Clone)]
struct PyTransition {
    inner: PpoTransition,
}

#[pymethods]
impl PyTransition {
    #[new]
    fn new(observation: [f32; 4], action: f32, log_prob: f32, reward: f32, done: f32) -> Self {
        Self {
            inner: PpoTransition {
                observation,
                action,
                log_prob,
                reward,
                done,
            },
        }
    }

    #[getter]
    fn observation(&self) -> [f32; 4] {
        self.inner.observation
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
            "Transition(obs={:?}, a={}, lp={:.3f}, r={}, done={})",
            self.inner.observation,
            self.inner.action,
            self.inner.log_prob,
            self.inner.reward,
            self.inner.done
        )
    }
}

/// Python-visible wrapper for the SimpleBuffer.
#[pyclass(name = "SimpleBuffer")]
struct PySimpleBuffer {
    buf: SimpleBuffer,
}

#[pymethods]
impl PySimpleBuffer {
    #[new]
    fn new(capacity: usize) -> Self {
        Self {
            buf: SimpleBuffer::new(capacity),
        }
    }

    /// Push one transition; overwrites oldest when full.
    fn write(&mut self, t: &PyTransition) {
        self.buf.write(t.inner);
    }

    /// Pull up to n transitions as a list of Transition objects.
    fn read_batch(&self, n: usize) -> Vec<PyTransition> {
        self.buf
            .read_batch(n)
            .into_iter()
            .map(|inner| PyTransition { inner })
            .collect()
    }

    #[getter]
    fn len(&self) -> usize {
        self.buf.len()
    }
    #[getter]
    fn capacity(&self) -> usize {
        self.buf.capacity()
    }
    #[getter]
    fn is_full(&self) -> bool {
        self.buf.is_full()
    }

    fn __repr__(&self) -> String {
        format!(
            "SimpleBuffer(len={}, capacity={}, full={})",
            self.buf.len(),
            self.buf.capacity(),
            self.buf.is_full()
        )
    }
}

// MUST be named `_core` — matches module-name in pyproject.toml
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTransition>()?;
    m.add_class::<PySimpleBuffer>()?;
    Ok(())
}
