use pyo3::prelude::*;

mod logging_utils;
mod configs;
mod results;
mod model;

/// A Python module implemented in Rust.
#[pymodule]
fn llama_rs_python(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_class::<model::Model>()?;

    m.add_class::<configs::GenerationConfig>()?;
    m.add_class::<configs::Precision>()?;
    m.add_class::<configs::SessionConfig>()?;

    m.add_class::<results::GenerationTimes>()?;
    m.add_class::<results::StopReason>()?;
    m.add_class::<results::GenerationResult>()?;

    Ok(())
}
