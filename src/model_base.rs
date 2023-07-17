use llm_base::InferenceParameters;
use llm_base::InferenceSession;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use llm::{InferenceError, TokenUtf8Buffer};
use llm_base::{InferenceFeedback, OutputRequest, Prompt};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use std::convert::Infallible;
use std::sync::Mutex;

use crate::configs;
use crate::configs::GenerationConfig;
use crate::results;
use std::sync::Arc;

#[pyclass]
pub struct GenerationStreamer {
    #[pyo3(get)]
    pub counter: usize,
    pub inference_params: Arc<InferenceParameters>,
    pub generation_config: Arc<Mutex<configs::GenerationConfig>>,
    pub session: Arc<Mutex<InferenceSession>>,
    pub rng: Arc<Mutex<ChaCha8Rng>>,
    pub model: Arc<dyn llm::Model>,
}

#[pymethods]
impl GenerationStreamer {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, _py: Python) -> Result<Option<String>, PyErr> {
        let mut output_request = OutputRequest::default();

        let inference_params = Arc::clone(&slf.inference_params);
        let generation_config_ref = Arc::clone(&slf.generation_config);
        let rng_ref = Arc::clone(&slf.rng);
        let model = Arc::clone(&slf.model);
        let session_ref = Arc::clone(&slf.session);
        let mut utf8_buf = TokenUtf8Buffer::new();

        loop {
            let generation_config = generation_config_ref.lock().unwrap();
            //Check if we have reached the maximum token count
            if generation_config.max_new_tokens.is_some()
                && slf.counter >= generation_config.max_new_tokens.unwrap()
            {
                return Ok(None);
            }
            //Forcefully drop the lock
            std::mem::drop(generation_config);

            let mut token: Option<String> = None;

            _py.allow_threads(|| {
                //Acquire locks in the _py context
                let mut session = session_ref.lock().unwrap();
                let mut rng = rng_ref.lock().unwrap();
                let mut generation_config = generation_config_ref.lock().unwrap();

                token = _infer_next_token(
                    &mut session,
                    &*model,
                    &inference_params,
                    &mut generation_config,
                    &mut output_request,
                    &mut rng,
                    &mut utf8_buf,
                )
                .unwrap()
            });

            if token.is_none() {
                return Ok(None);
            }
            //Increase counter
            slf.counter += 1;

            if let Some(token) = token {
                return Ok(Some(token));
            }
        }
    }
}

pub fn _tokenize(model: &dyn llm::Model, text: &str) -> Result<Vec<u32>, InferenceError> {
    Ok(model
        .tokenizer()
        .tokenize(text, false)?
        .iter()
        .map(|(_, token)| *token)
        .collect())
}

pub fn _decode(model: &dyn llm::Model, tokens: Vec<u32>) -> Result<String, std::str::Utf8Error> {
    let vocab = model.tokenizer();
    let characters: Vec<u8> = vocab.decode(tokens, false);

    match std::str::from_utf8(&characters) {
        Ok(text) => Ok(text.to_string()),
        Err(e) => Err(e),
    }
}

pub fn _start_session<'a>(
    model: &'a dyn llm::Model,
    session_config: &'a configs::SessionConfig,
    prompt: &'a str,
    generation_config: Option<configs::GenerationConfig>,
) -> (
    configs::GenerationConfig,
    InferenceParameters,
    ChaCha8Rng,
    Prompt<'a>,
    InferenceSession,
) {
    let session_params = session_config.to_llm_params();
    //Build the correct generation parameters
    let mut config_to_use = generation_config.unwrap_or(configs::GenerationConfig::default());

    let generation_params = config_to_use.to_llm_params();

    let rng = ChaCha8Rng::seed_from_u64(config_to_use.seed);
    let prompt = Prompt::from(prompt);

    config_to_use.init_stop_words(model);

    let session = model.start_session(session_params);
    (config_to_use, generation_params, rng, prompt, session)
}

pub fn _infer_next_token(
    session: &mut InferenceSession,
    model: &dyn llm::Model,
    infernece_params: &InferenceParameters,
    generation_params: &mut GenerationConfig,
    output_request: &mut OutputRequest,
    rng: &mut ChaCha8Rng,
    utf8_buf: &mut TokenUtf8Buffer,
) -> Result<Option<String>, PyErr> {
    loop {
        let token = match session.infer_next_token(model, infernece_params, output_request, rng) {
            Ok(token) => token,
            Err(InferenceError::EndOfText) => return Ok(None),
            Err(e) => return Err(PyException::new_err(e.to_string())),
        };

        //Check if the token is a stop word
        if let Some(stop_word_handler) = &mut generation_params.stop_word_handler {
            if stop_word_handler.process(token.to_vec()) {
                return Ok(None);
            }
        }

        //Buffer until a valid utf8 sequence is found
        if let Some(s) = utf8_buf.push(&token) {
            return Ok(Some(s));
        }
    }
}

pub fn _generate(
    _py: Python,
    model: &dyn llm::Model,
    session_config: &configs::SessionConfig,
    prompt: String,
    generation_config: Option<configs::GenerationConfig>,
    callback: Option<PyObject>,
) -> Result<results::GenerationResult, PyErr> {
    let (mut generation_config, inference_params, mut rng, prompt, mut session) =
        _start_session(model, session_config, &prompt, generation_config);

    //Extract the callback function from the python object
    let mut callback_function: Option<&PyAny> = None;
    let pytohn_object: Py<PyAny>;

    if let Some(unwrapped) = callback {
        pytohn_object = unwrapped;
        let python_function = pytohn_object.as_ref(_py);
        callback_function = Some(python_function);
        assert!(python_function.is_callable(), "Callback is not callable!");
    }

    let feed_start_at = std::time::SystemTime::now();
    //Feed the prompt

    let mut output_request_feeding = OutputRequest::default();
    _py.allow_threads(|| {
        session
            .feed_prompt::<Infallible, _>(model, prompt, &mut output_request_feeding, |_| {
                Ok(InferenceFeedback::Continue)
            })
            .unwrap()
    });
    let feed_prompt_duration = feed_start_at.elapsed().unwrap();

    //Start the inference loop
    let mut tokens_processed = 0;
    let mut token_utf8_buf = TokenUtf8Buffer::new();

    let mut total_generated_text = String::new();
    let stop_reason;

    let generation_start_at = std::time::SystemTime::now();

    let mut output_request_generation = OutputRequest::default();

    loop {
        //Check if we have reached the maximum token count
        if generation_config.max_new_tokens.is_some()
            && tokens_processed >= generation_config.max_new_tokens.unwrap()
        {
            stop_reason = results::StopReason::MaxLength;
            break;
        }

        //Infere the next token and break if the END_OF_TEXT token is reached
        let mut token: Option<String> = None;
        _py.allow_threads(|| {
            token = _infer_next_token(
                &mut session,
                model,
                &inference_params,
                &mut generation_config,
                &mut output_request_generation,
                &mut rng,
                &mut token_utf8_buf,
            )
            .unwrap()
        });

        if token.is_none() {
            stop_reason = results::StopReason::EndToken;
            break;
        }

        if let Some(tokens) = token {
            total_generated_text.push_str(&tokens);
            //Check if the callback function is set and call it
            if let Some(function) = callback_function {
                let args = PyTuple::new(_py, &[tokens]);
                let result = function.call1(args)?;
                //Check if the callback function returned true
                if !result.is_none() && result.is_true()? {
                    //If the callback function returned true, stop the generation
                    stop_reason = results::StopReason::UserCancelled;
                    break;
                }
            }
        }

        tokens_processed += 1;
    }

    let generation_duration = generation_start_at.elapsed().unwrap();

    let result = results::GenerationResult {
        text: total_generated_text,
        times: results::GenerationTimes {
            total: generation_duration.as_millis() + feed_prompt_duration.as_millis(),
            generation: generation_duration.as_millis(),
            prompt_feeding: feed_prompt_duration.as_millis(),
        },
        stop_reason,
    };

    Ok(result)
}

pub fn _embed(
    _py: Python,
    model: &dyn llm::Model,
    session_config: &configs::SessionConfig,
    prompt: String,
) -> Result<Vec<f32>, PyErr> {
    let (_, _, _, prompt, mut session) = _start_session(model, session_config, &prompt, None);

    //Feed the prompt
    let mut output_request_feeding = OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    _py.allow_threads(|| {
        session
            .feed_prompt::<Infallible, _>(model, prompt, &mut output_request_feeding, |_| {
                Ok(InferenceFeedback::Continue)
            })
            .unwrap()
    });
    Ok(output_request_feeding.embeddings.unwrap())
}

macro_rules! wrap_model {
    ($name:ident,$llm_model:ty) => {
        #[pyclass]
        #[allow(clippy::upper_case_acronyms)]
        pub struct $name {
            #[pyo3(get, set)]
            pub config: crate::configs::SessionConfig,
            #[pyo3(get)]
            pub verbose: bool,
            #[pyo3(get)]
            pub path: String,
            #[pyo3(get)]
            pub lora_paths: Option<Vec<String>>,
            pub llm_model: std::sync::Arc<dyn llm::Model>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new(
                path: String,
                session_config: Option<crate::configs::SessionConfig>,
                tokenizer_name_or_path: Option<String>,
                lora_paths: Option<Vec<String>>,
                verbose: Option<bool>,
            ) -> Self {
                let should_log = verbose.unwrap_or(false);

                //Build the correct session parameters
                let default_config = crate::configs::SessionConfig::default();
                let config_to_use = session_config.unwrap_or(default_config);

                let path = std::path::Path::new(&path);
                let lora_paths = lora_paths
                    .map(|strings| strings.into_iter().map(std::path::PathBuf::from).collect());
                let model_params = llm_base::ModelParameters {
                    context_size: config_to_use.context_length,
                    prefer_mmap: config_to_use.prefer_mmap,
                    lora_adapters: lora_paths.clone(),
                    use_gpu: config_to_use.use_gpu,
                    gpu_layers: config_to_use.gpu_layers,
                };

                let vocabulary_source: llm_base::TokenizerSource;

                if let Some(name_or_path) = tokenizer_name_or_path {
                    let tokenizer_path = std::path::Path::new(&name_or_path);
                    if tokenizer_path.is_file() && tokenizer_path.exists() {
                        // Load tokenizer from file
                        vocabulary_source = llm_base::TokenizerSource::HuggingFaceTokenizerFile(
                            tokenizer_path.to_owned(),
                        );
                    } else {
                        // Load tokenizer from HuggingFace
                        vocabulary_source =
                            llm_base::TokenizerSource::HuggingFaceRemote(name_or_path);
                    }
                } else {
                    vocabulary_source = llm_base::TokenizerSource::Embedded;
                }

                let llm_model: $llm_model =
                    llm_base::load(&path, vocabulary_source, model_params, |load_progress| {
                        if should_log {
                            llm_base::load_progress_callback_stdout(load_progress)
                        }
                    })
                    .unwrap();

                $name {
                    config: config_to_use,
                    verbose: should_log,
                    path: path.to_str().unwrap().to_string(),
                    llm_model: std::sync::Arc::new(llm_model),
                    lora_paths: lora_paths.map(|paths| {
                        paths
                            .into_iter()
                            .map(|p| p.to_str().unwrap().to_string())
                            .collect()
                    }),
                }
            }

            fn generate(
                &self,
                _py: Python,
                prompt: String,
                generation_config: Option<crate::configs::GenerationConfig>,
                callback: Option<PyObject>,
            ) -> PyResult<crate::results::GenerationResult> {
                return crate::model_base::_generate(
                    _py,
                    self.llm_model.as_ref(),
                    &self.config,
                    prompt,
                    generation_config,
                    callback,
                );
            }

            fn embed(&self, _py: Python, prompt: String) -> PyResult<Vec<f32>> {
                return crate::model_base::_embed(
                    _py,
                    self.llm_model.as_ref(),
                    &self.config,
                    prompt,
                );
            }

            fn stream(
                &self,
                _py: Python,
                prompt: String,
                generation_config: Option<crate::configs::GenerationConfig>,
            ) -> PyResult<crate::model_base::GenerationStreamer> {
                let (generation_config, inference_params, rng, prompt, mut session) =
                    crate::model_base::_start_session(
                        self.llm_model.as_ref(),
                        &self.config,
                        &prompt,
                        generation_config,
                    );

                //feed the session
                let mut output_request_feeding = llm_base::OutputRequest::default();
                _py.allow_threads(|| {
                    session
                        .feed_prompt::<std::convert::Infallible, _>(
                            self.llm_model.as_ref(),
                            prompt,
                            &mut output_request_feeding,
                            |_| Ok(llm_base::InferenceFeedback::Continue),
                        )
                        .unwrap()
                });

                Ok(crate::model_base::GenerationStreamer {
                    generation_config: std::sync::Arc::new(std::sync::Mutex::new(
                        generation_config,
                    )),
                    inference_params: std::sync::Arc::new(inference_params),
                    session: std::sync::Arc::new(std::sync::Mutex::new(session)),
                    rng: std::sync::Arc::new(std::sync::Mutex::new(rng)),
                    model: self.llm_model.clone(),
                    counter: 0,
                })
            }

            fn tokenize(&self, text: String) -> PyResult<Vec<u32>> {
                match crate::model_base::_tokenize(self.llm_model.as_ref(), &text) {
                    Ok(tokens) => Ok(tokens),
                    Err(e) => Err(pyo3::exceptions::PyException::new_err(e.to_string())),
                }
            }

            fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
                match crate::model_base::_decode(self.llm_model.as_ref(), tokens) {
                    Ok(tokens) => Ok(tokens),
                    Err(e) => Err(pyo3::exceptions::PyException::new_err(e.to_string())),
                }
            }

            #[staticmethod]
            fn quantize(
                _py: Python,
                source: String,
                destination: String,
                quantization: Option<crate::quantize::QuantizationType>,
                container: Option<crate::quantize::ContainerType>,
                callback: Option<PyObject>,
            ) -> PyResult<()> {
                let mut callback_function: Option<&PyAny> = None;
                let pytohn_object: Py<PyAny>;

                if let Some(unwrapped) = callback {
                    pytohn_object = unwrapped;
                    let python_function = pytohn_object.as_ref(_py);
                    callback_function = Some(python_function);
                    assert!(python_function.is_callable(), "Callback is not callable!");
                }

                crate::quantize::_quantize::<$llm_model>(
                    source.into(),
                    destination.into(),
                    container.unwrap_or(crate::quantize::ContainerType::GGJT),
                    quantization.unwrap_or(crate::quantize::QuantizationType::Q4_0),
                    |message| {
                        if let Some(callback) = callback_function {
                            let args = pyo3::types::PyTuple::new(_py, &[message]);
                            callback.call1(args).unwrap();
                        }
                    },
                )
                .map_err(|e| pyo3::exceptions::PyException::new_err(e.to_string()))
            }
        }
    };
}

pub(crate) use wrap_model;
