use pyo3::prelude::*;

mod configs;
mod model_base;
mod models;
mod quantize;
mod results;

#[pymodule]
fn llm_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    let config_module = PyModule::new(_py, "config")?;
    config_module.add_class::<configs::GenerationConfig>()?;
    config_module.add_class::<configs::Precision>()?;
    config_module.add_class::<configs::SessionConfig>()?;
    config_module.add_class::<quantize::ContainerType>()?;
    config_module.add_class::<quantize::QuantizationType>()?;
    m.add_submodule(config_module)?;

    let results_module = PyModule::new(_py, "results")?;
    results_module.add_class::<results::GenerationTimes>()?;
    results_module.add_class::<results::StopReason>()?;
    results_module.add_class::<results::GenerationResult>()?;
    m.add_submodule(results_module)?;

    let models_module = PyModule::new(_py, "models")?;
    models_module.add_class::<models::Llama>()?;
    models_module.add_class::<models::GptJ>()?;
    models_module.add_class::<models::Gpt2>()?;
    models_module.add_class::<models::GptNeoX>()?;
    models_module.add_class::<models::Bloom>()?;
    models_module.add_class::<models::Mpt>()?;
    m.add_submodule(models_module)?;

    //hacky but apparently the only way to get a submodule to work on all platforms with typehints
    //see https://github.com/PyO3/pyo3/issues/759
    _py.import("sys")?
        .getattr("modules")?
        .set_item("llm_rs.config", config_module)?;

    _py.import("sys")?
        .getattr("modules")?
        .set_item("llm_rs.results", results_module)?;

    _py.import("sys")?
        .getattr("modules")?
        .set_item("llm_rs.models", models_module)?;

    Ok(())
}
