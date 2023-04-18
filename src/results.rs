use pyo3::prelude::*;


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

#[pyclass]
pub struct GenerationResult {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub times: GenerationTimes,
}