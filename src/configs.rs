use pyo3::prelude::*;
use llama_rs::{InferenceParameters,TokenBias};

#[pyclass]
pub struct GenerationConfig {
    #[pyo3(get, set)]
    top_k:usize,
    #[pyo3(get, set)]
    top_p:f32,
    #[pyo3(get, set)]
    temperature:f32,
    #[pyo3(get, set)]
    repetition_penalty:f32,
}

#[pymethods]
impl GenerationConfig {

    #[new]
    fn new(top_k: Option<usize>,top_p:Option<f32>,temperature:Option<f32>,repetition_penalty:Option<f32>) -> Self {
        GenerationConfig {
            top_k: top_k.unwrap_or(40),
            top_p: top_p.unwrap_or(0.95),
            temperature: temperature.unwrap_or(0.80),
            repetition_penalty: repetition_penalty.unwrap_or(1.30),
        }
    }
}

impl GenerationConfig{
    pub fn to_llama_rs_params(&self,n_threads:usize,n_batch:usize)->InferenceParameters{
        InferenceParameters{
            top_k: self.top_k,
            top_p: self.top_p,
            temperature: self.temperature,
            repeat_penalty: self.repetition_penalty,
            bias_tokens: TokenBias::default(),
            play_back_previous_tokens: false,
            n_threads: n_threads,
            n_batch: n_batch,
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            top_k:40,
            top_p:0.95,
            temperature:0.80,
            repetition_penalty:1.30,
        }
    }
}

