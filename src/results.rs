use pyo3::prelude::*;
use std::fmt;

#[pyclass]
#[derive(Clone)]
pub struct GenerationTimes {
    #[pyo3(get)]
    pub total: u128,
    #[pyo3(get)]
    pub generation: u128,
    #[pyo3(get)]
    pub prompt_feeding: u128,
}

impl fmt::Display for GenerationTimes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(Total: {} ms, Feeding: {} ms, Generation: {} ms)",
            self.total, self.prompt_feeding, self.generation
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub enum StopReason {
    EndToken,
    MaxLength,
    UserCancelled,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StopReason::EndToken => write!(f, "EndToken"),
            StopReason::MaxLength => write!(f, "MaxLength"),
            StopReason::UserCancelled => write!(f, "UserCancelled"),
        }
    }
}

#[pyclass]
pub struct GenerationResult {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub times: GenerationTimes,
    #[pyo3(get)]
    pub stop_reason: StopReason,
}

#[pymethods]
impl GenerationResult {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "GenerationResult(text='{}', times={}, stop_reason={})",
            self.text, self.times, self.stop_reason
        ))
    }
}
