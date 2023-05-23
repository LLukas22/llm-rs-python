use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use llm::{InferenceError, TokenUtf8Buffer};
use llm_base::{InferenceFeedback, OutputRequest, Prompt};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use std::convert::Infallible;

use crate::configs;
use crate::results;

pub fn _tokenize(model: &dyn llm::Model, text: &str) -> Result<Vec<i32>, InferenceError> {
    Ok(model
        .vocabulary()
        .tokenize(text, false)?
        .iter()
        .map(|(_, token)| *token)
        .collect())
}

pub fn _decode(model: &dyn llm::Model, tokens: Vec<i32>) -> Result<String, std::str::Utf8Error> {
    let vocab = model.vocabulary();
    let characters: Vec<u8> = tokens
        .into_iter()
        .flat_map(|token| vocab.id_to_token[token as usize].to_owned())
        .collect();

    match std::str::from_utf8(&characters) {
        Ok(text) => Ok(text.to_string()),
        Err(e) => Err(e),
    }
}

pub fn _generate(
    _py: Python,
    model: &dyn llm::Model,
    session_config: &configs::SessionConfig,
    prompt: String,
    generation_config: Option<&configs::GenerationConfig>,
    callback: Option<PyObject>,
) -> Result<results::GenerationResult, PyErr> {
    let session_params = session_config.to_llm_params();

    //Build the correct generation parameters
    let default_config = configs::GenerationConfig::default();
    let config_to_use = generation_config.unwrap_or(&default_config);

    let generation_params =
        config_to_use.to_llm_params(session_config.threads, session_config.batch_size);

    let mut rng = ChaCha8Rng::seed_from_u64(config_to_use.seed);
    let prompt = Prompt::from(&prompt);

    let mut session = model.start_session(session_params);

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
            .feed_prompt::<Infallible, _>(
                model,
                &generation_params,
                prompt,
                &mut output_request_feeding,
                |_| Ok(InferenceFeedback::Continue),
            )
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
    let mut _generate_next_tokens = || -> Result<Option<String>, PyErr> {
        let token = match session.infer_next_token(
            model,
            &generation_params,
            &mut output_request_generation,
            &mut rng,
        ) {
            Ok(token) => token,
            Err(InferenceError::EndOfText) => return Ok(None),
            Err(e) => return Err(PyException::new_err(e.to_string())),
        };
        Ok(token_utf8_buf.push(token))
    };

    loop {
        //Check if we have reached the maximum token count
        if config_to_use.max_new_tokens.is_some()
            && tokens_processed >= config_to_use.max_new_tokens.unwrap()
        {
            stop_reason = results::StopReason::MaxLength;
            break;
        }

        //Infere the next token and break if the END_OF_TEXT token is reached

        let mut token: Option<String> = None;
        _py.allow_threads(|| token = _generate_next_tokens().unwrap());

        if token.is_none() {
            stop_reason = results::StopReason::EndToken;
            break;
        }
        //Buffer until a valid utf8 sequence is found
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
            pub llm_model: Box<dyn llm::Model>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new(
                path: String,
                session_config: Option<crate::configs::SessionConfig>,
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
                    ..Default::default()
                };
                let llm_model: $llm_model =
                    llm_base::load(&path, model_params, None, |load_progress| {
                        if should_log {
                            llm_base::load_progress_callback_stdout(load_progress)
                        }
                    })
                    .unwrap();

                $name {
                    config: config_to_use,
                    verbose: should_log,
                    path: path.to_str().unwrap().to_string(),
                    llm_model: Box::new(llm_model),
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
                generation_config: Option<&crate::configs::GenerationConfig>,
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

            fn tokenize(&self, text: String) -> PyResult<Vec<i32>> {
                match crate::model_base::_tokenize(self.llm_model.as_ref(), &text) {
                    Ok(tokens) => Ok(tokens),
                    Err(e) => Err(pyo3::exceptions::PyException::new_err(e.to_string())),
                }
            }

            fn decode(&self, tokens: Vec<i32>) -> PyResult<String> {
                match crate::model_base::_decode(self.llm_model.as_ref(), tokens) {
                    Ok(tokens) => Ok(tokens),
                    Err(e) => Err(pyo3::exceptions::PyException::new_err(e.to_string())),
                }
            }

            #[staticmethod]
            fn quantize(
                source: String,
                destination: String,
                quantization: Option<crate::quantize::QuantizationType>,
                container: Option<crate::quantize::ContainerType>,
            ) -> PyResult<()> {
                crate::quantize::_quantize::<$llm_model>(
                    source.into(),
                    destination.into(),
                    container.unwrap_or(crate::quantize::ContainerType::GGJT),
                    quantization.unwrap_or(crate::quantize::QuantizationType::Q4_0),
                )
                .map_err(|e| pyo3::exceptions::PyException::new_err(e.to_string()))
            }
        }
    };
}

pub(crate) use wrap_model;
