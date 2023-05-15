use llm::{InferenceParameters, InferenceSessionConfig, ModelKVMemoryType, TokenBias};
use pyo3::prelude::*;

#[pyclass]
pub struct GenerationConfig {
    #[pyo3(get, set)]
    pub top_k: usize,
    #[pyo3(get, set)]
    pub top_p: f32,
    #[pyo3(get, set)]
    pub temperature: f32,
    #[pyo3(get, set)]
    pub repetition_penalty: f32,
    #[pyo3(get, set)]
    pub repetition_penalty_last_n: usize,
    #[pyo3(get, set)]
    pub seed: u64,
    #[pyo3(get, set)]
    pub max_new_tokens: Option<usize>,
    #[pyo3(get, set)]
    pub stop_words: Option<Vec<String>>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            top_k: 40,
            top_p: 0.95,
            temperature: 0.80,
            repetition_penalty: 1.30,
            repetition_penalty_last_n: 512,
            seed: 42,
            max_new_tokens: None,
            stop_words: None,
        }
    }
}

#[pymethods]
impl GenerationConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        top_k: Option<usize>,
        top_p: Option<f32>,
        temperature: Option<f32>,
        repetition_penalty: Option<f32>,
        repetition_penalty_last_n: Option<usize>,
        seed: Option<u64>,
        max_new_tokens: Option<usize>,
        stop_words: Option<Vec<String>>,
    ) -> Self {
        GenerationConfig {
            top_k: top_k.unwrap_or(40),
            top_p: top_p.unwrap_or(0.95),
            temperature: temperature.unwrap_or(0.80),
            repetition_penalty: repetition_penalty.unwrap_or(1.30),
            repetition_penalty_last_n: repetition_penalty_last_n.unwrap_or(512),
            seed: seed.unwrap_or(42),
            max_new_tokens,
            stop_words,
        }
    }
}

impl GenerationConfig {
    pub fn to_llm_params(&self, n_threads: usize, n_batch: usize) -> InferenceParameters {
        InferenceParameters {
            top_k: self.top_k,
            top_p: self.top_p,
            temperature: self.temperature,
            repeat_penalty: self.repetition_penalty,
            repetition_penalty_last_n: self.repetition_penalty_last_n,
            bias_tokens: TokenBias::default(),
            n_threads,
            n_batch,
        }
    }
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Precision {
    FP32,
    FP16,
}

impl Precision {
    pub fn to_llama_rs_memory_type(self) -> ModelKVMemoryType {
        match self {
            Precision::FP16 => ModelKVMemoryType::Float16,
            Precision::FP32 => ModelKVMemoryType::Float32,
        }
    }
}

#[pyclass]
#[derive(Clone, Copy)]
pub struct SessionConfig {
    #[pyo3(get, set)]
    pub threads: usize,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get)]
    pub context_length: usize,
    #[pyo3(get, set)]
    pub keys_memory_type: Precision,
    #[pyo3(get, set)]
    pub values_memory_type: Precision,
    #[pyo3(get)]
    pub prefer_mmap: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            threads: 8,
            batch_size: 8,
            context_length: 2048,
            keys_memory_type: Precision::FP32,
            values_memory_type: Precision::FP32,
            prefer_mmap: true,
        }
    }
}

#[pymethods]
impl SessionConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        threads: Option<usize>,
        batch_size: Option<usize>,
        context_length: Option<usize>,
        keys_memory_type: Option<Precision>,
        values_memory_type: Option<Precision>,
        prefer_mmap: Option<bool>,
    ) -> Self {
        SessionConfig {
            threads: threads.unwrap_or(8),
            batch_size: batch_size.unwrap_or(8),
            context_length: context_length.unwrap_or(2048),
            keys_memory_type: keys_memory_type.unwrap_or(Precision::FP32),
            values_memory_type: values_memory_type.unwrap_or(Precision::FP32),
            prefer_mmap: prefer_mmap.unwrap_or(true),
        }
    }
}

impl SessionConfig {
    pub fn to_llm_params(self) -> InferenceSessionConfig {
        InferenceSessionConfig {
            memory_k_type: self.keys_memory_type.to_llama_rs_memory_type(),
            memory_v_type: self.values_memory_type.to_llama_rs_memory_type(),
        }
    }
}
