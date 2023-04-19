use pyo3::prelude::*;

use llama_rs::{InferenceError, TokenUtf8Buffer};
use pyo3::exceptions::PyException;
use pyo3::types::{PyTuple};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::configs;
use crate::results;
use crate::logging_utils;

#[pyclass]
pub struct Model {
    model: llama_rs::Model,
    #[pyo3(get, set)]
    config: configs::SessionConfig,
    #[pyo3(get, set)]
    verbose: bool,
    #[pyo3(get)]
    path: String,
}

//Ok im not even pretending to know what this does
unsafe impl Send for Model {}

#[pymethods]
impl Model {
    
    #[new]
    fn new(path: String, session_config:Option<configs::SessionConfig>,verbose:Option<bool>) -> Self {
        let should_log = verbose.unwrap_or(false);
        
        //Build the correct session parameters
        let default_config = configs::SessionConfig::default();
        let config_to_use = session_config.unwrap_or(default_config);

        //Load the model
        let model = llama_rs::Model::load(
            &path,
            config_to_use.context_length,
            |load_progress| {
            if should_log{
                logging_utils::log_load_progress(load_progress)
            }
        }).unwrap();
        Model { model: model, config:config_to_use, verbose: should_log, path:path }
    } 

    fn generate(
        &self,
        _py: Python,
        prompt: String,
        generation_config: Option<&configs::GenerationConfig>,
        callback: Option<PyObject>,
    ) -> PyResult<results::GenerationResult> {

        let session_params = self.config.to_llama_rs_params();
        
        //Build the correct generation parameters
        let default_config = configs::GenerationConfig::default();
        let config_to_use = generation_config.unwrap_or(&default_config);

        let generation_params = config_to_use.to_llama_rs_params(self.config.threads,  self.config.batch_size);

        let mut rng = ChaCha8Rng::seed_from_u64(config_to_use.seed);
        let prompt = prompt.as_str();

        let mut session = self.model.start_session(session_params);

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
        session.feed_prompt(&self.model, &generation_params, &prompt, |_| {
            Ok::<(), InferenceError>(())
        });
        let feed_prompt_duration = feed_start_at.elapsed().unwrap();


        //Start the inference loop
        let mut tokens_processed = 0;
        let mut token_utf8_buf = TokenUtf8Buffer::new();

        let mut total_generated_text = String::new();
        let mut stop_reason = results::StopReason::EndToken;

        let generation_start_at = std::time::SystemTime::now();
        loop {
            //Check if we have reached the maximum token count
            if config_to_use.max_new_tokens.is_some() && tokens_processed >= config_to_use.max_new_tokens.unwrap() {
                stop_reason=results::StopReason::MaxLength;
                break;
            }

            //Infere the next token and break if the END_OF_TEXT token is reached
            let token = match session.infer_next_token(&self.model, &generation_params, &mut rng) {
                Ok(token) => token,
                Err(InferenceError::EndOfText) => break,
                Err(e) => return Err(PyException::new_err(e.to_string())),
            };

            //Buffer until a valid utf8 sequence is found
            if let Some(tokens) = token_utf8_buf.push(token) {

                total_generated_text.push_str(&tokens);
                //Check if the callback function is set and call it
                if let Some(function) = callback_function {
                    let args = PyTuple::new(_py, &[tokens]);
                    let result = function.call1(args)?;
                    //Check if the callback function returned true
                    if !result.is_none() {
                        if result.is_true()? {
                            //If the callback function returned true, stop the generation
                            stop_reason=results::StopReason::UserCancelled;
                            break;
                        }
                    }
                }
            }

            tokens_processed += 1;
        }
        let generation_duration = generation_start_at.elapsed().unwrap();

        let result = results::GenerationResult{
            text:total_generated_text,
            times: results::GenerationTimes{
                total:generation_duration.as_millis()+feed_prompt_duration.as_millis(),
                generation:generation_duration.as_millis(),
                prompt_feeding:feed_prompt_duration.as_millis()},
            stop_reason:stop_reason
            };

        Ok(result)
    }
}