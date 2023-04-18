use pyo3::prelude::*;

use llama_rs::{InferenceError, InferenceSessionParameters, TokenUtf8Buffer};
use pyo3::exceptions::PyException;
use pyo3::types::{PyTuple};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

mod logging_utils;
mod configs;
mod results;

#[pyclass]
pub struct GenerationResult {}

#[pyclass]
#[pyo3(text_signature = "(path, context_length=2048, threads=8, batch_size=8, verbose=False,/)")]
pub struct Model {
    model: llama_rs::Model,
    #[pyo3(get, set)]
    batch_size: usize,
    #[pyo3(get, set)]
    threads: usize,
    #[pyo3(get, set)]
    verbose: bool,
}

//Ok im not even pretending to know what this does
unsafe impl Send for Model {}

#[pymethods]
impl Model {
    
    #[new]
    fn new(path: String, context_length: Option<usize>,threads:Option<usize>,batch_size:Option<usize>,verbose:Option<bool>) -> Self {
        let should_log = verbose.unwrap_or(false);
        let model = llama_rs::Model::load(
            &path,
            context_length.unwrap_or(2048),
            |load_progress| {
            if should_log{
                logging_utils::log_load_progress(load_progress)
            }
        }).unwrap();
        Model { model: model, batch_size: batch_size.unwrap_or(8), threads: threads.unwrap_or(8), verbose: should_log }
    } 

    #[pyo3(text_signature = "(prompt, seed=42, maximum_token_count=256, callback=None, /)")]
    fn generate(
        &self,
        _py: Python,
        prompt: String,
        generation_config: Option<&configs::GenerationConfig>,
        seed: Option<u64>,
        maximum_token_count: Option<usize>,
        callback: Option<PyObject>,
    ) -> PyResult<results::GenerationResult> {

        let session_params = InferenceSessionParameters::default();
        
        //Build the correct generation parameters
        let default_config = configs::GenerationConfig::default();
        let config_to_use = generation_config.unwrap_or(&default_config);

        let generation_params = config_to_use.to_llama_rs_params(self.threads, self.batch_size);

        let mut session = self.model.start_session(session_params);

        let mut rng = ChaCha8Rng::seed_from_u64(seed.unwrap_or(42));
        let prompt = prompt.as_str();

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

        let generation_start_at = std::time::SystemTime::now();
        loop {
            //Check if we have reached the maximum token count
            if maximum_token_count.is_some() && tokens_processed >= maximum_token_count.unwrap() {
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
                prompt_feeding:feed_prompt_duration.as_millis()}
            };

        Ok((result))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn llama_rs_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<configs::GenerationConfig>()?;
    m.add_class::<results::GenerationTimes>()?;
    m.add_class::<results::GenerationResult>()?;
    Ok(())
}
