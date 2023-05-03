use pyo3::prelude::*;

mod configs;
mod results;
mod models;
mod model_base;




#[pymodule]
fn llm_rs(_py: Python, m: &PyModule) -> PyResult<()> {

    let config_module = PyModule::new(_py, "config")?;
    config_module.add_class::<configs::GenerationConfig>()?;
    config_module.add_class::<configs::Precision>()?;
    config_module.add_class::<configs::SessionConfig>()?;
    m.add_submodule(config_module)?;

    let results_module = PyModule::new(_py, "results")?;
    results_module.add_class::<results::GenerationTimes>()?;
    results_module.add_class::<results::StopReason>()?;
    results_module.add_class::<results::GenerationResult>()?;
    m.add_submodule(results_module)?;

    m.add_class::<model_base::InternalModel>()?;
    m.add_class::<models::Llama>()?;
    m.add_class::<models::GPTJ>()?;
    
    
    //hacky but apparently the only way to get a submodule to work on all platforms with typehints
    //see https://github.com/PyO3/pyo3/issues/759
    _py.import("sys")?
        .getattr("modules")?
        .set_item("llm_rs.config", config_module)?;

    _py.import("sys")?
        .getattr("modules")?
        .set_item("llm_rs.results", results_module)?;

    Ok(())
}
