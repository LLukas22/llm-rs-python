use crate::stopwords::StopWordHandler;
use llm::{InferenceParameters, InferenceSessionConfig, ModelKVMemoryType, TokenBias};
use pyo3::{prelude::*, types::PyBytes};
use serde::{Deserialize, Serialize};

#[pyclass]
#[pyo3(module = "llm_rs.config")]
#[derive(Clone, Serialize, Deserialize)]
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
    pub stop_word_handler: Option<StopWordHandler>,
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
            stop_word_handler: None,
        }
    }
}

impl GenerationConfig {
    pub fn init_stop_words(&mut self, model: &dyn llm::Model) {
        if self.stop_words.is_some() {
            let stopwords = self.stop_words.clone().unwrap();
            self.stop_word_handler = Some(StopWordHandler::new(model, &stopwords));
        } else {
            self.stop_word_handler = None;
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
            stop_word_handler: None,
        }
    }

    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = serde_json::from_slice(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        Ok(PyBytes::new(py, &serde_json::to_vec(&self).unwrap()))
    }

    #[allow(clippy::type_complexity)]
    pub fn __getnewargs__(
        &self,
    ) -> PyResult<(
        usize,
        f32,
        f32,
        f32,
        usize,
        u64,
        Option<usize>,
        Option<Vec<String>>,
    )> {
        Ok((
            self.top_k,
            self.top_p,
            self.temperature,
            self.repetition_penalty,
            self.repetition_penalty_last_n,
            self.seed,
            self.max_new_tokens,
            self.stop_words.clone(),
        ))
    }
}

impl GenerationConfig {
    pub fn to_llm_params(&self) -> InferenceParameters {
        InferenceParameters {
            sampler: std::sync::Arc::new(llm::samplers::TopPTopK {
                top_k: self.top_k,
                top_p: self.top_p,
                temperature: self.temperature,
                repeat_penalty: self.repetition_penalty,
                repetition_penalty_last_n: self.repetition_penalty_last_n,
                bias_tokens: TokenBias::default(),
            }),
        }
    }
}

#[pyclass]
#[pyo3(module = "llm_rs.config")]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Precision {
    FP32,
    FP16,
}

#[pymethods]
impl Precision {
    #[new]
    fn new(value: usize) -> Self {
        match value {
            16 => Precision::FP16,
            32 => Precision::FP32,
            _ => panic!("Invalid precision value"),
        }
    }

    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = serde_json::from_slice(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        Ok(PyBytes::new(py, &serde_json::to_vec(&self).unwrap()))
    }

    pub fn __getnewargs__(&self) -> PyResult<(usize,)> {
        //Hack to get pyo3 enum pickle serializable
        match self {
            Precision::FP16 => Ok((16_usize,)),
            Precision::FP32 => Ok((32_usize,)),
        }
    }
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
#[pyo3(module = "llm_rs.config")]
#[derive(Clone, Copy, Serialize, Deserialize)]
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
    #[pyo3(get)]
    pub use_gpu: bool,
    #[pyo3(get)]
    pub gpu_layers: Option<usize>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            threads: 8,
            batch_size: 8,
            context_length: 2048,
            keys_memory_type: Precision::FP16,
            values_memory_type: Precision::FP16,
            prefer_mmap: true,
            use_gpu: false,
            gpu_layers: None,
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
        use_gpu: Option<bool>,
        gpu_layers: Option<usize>,
    ) -> Self {
        SessionConfig {
            threads: threads.unwrap_or(8),
            batch_size: batch_size.unwrap_or(8),
            context_length: context_length.unwrap_or(2048),
            keys_memory_type: keys_memory_type.unwrap_or(Precision::FP16),
            values_memory_type: values_memory_type.unwrap_or(Precision::FP16),
            prefer_mmap: prefer_mmap.unwrap_or(true),
            use_gpu: use_gpu.unwrap_or(false),
            gpu_layers,
        }
    }

    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = serde_json::from_slice(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        Ok(PyBytes::new(py, &serde_json::to_vec(&self).unwrap()))
    }
    #[allow(clippy::type_complexity)]
    pub fn __getnewargs__(
        &self,
    ) -> PyResult<(usize, usize, usize, Precision, Precision, bool, bool, usize)> {
        Ok((
            self.threads,
            self.batch_size,
            self.context_length,
            self.keys_memory_type,
            self.values_memory_type,
            self.prefer_mmap,
            self.use_gpu,
            self.gpu_layers.unwrap_or(0),
        ))
    }
}

impl SessionConfig {
    pub fn to_llm_params(self) -> InferenceSessionConfig {
        InferenceSessionConfig {
            memory_k_type: self.keys_memory_type.to_llama_rs_memory_type(),
            memory_v_type: self.values_memory_type.to_llama_rs_memory_type(),
            n_batch: self.batch_size,
            n_threads: self.threads,
        }
    }
}
