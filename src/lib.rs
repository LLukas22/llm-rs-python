use pyo3::prelude::*;
use llama_rs::{Model, InferenceSessionParameters,InferenceParameters};
use std::{path::Path, default};
use rand::{prelude::*, Error};

#[pyclass]
pub struct LlamaModel{
    model: Model
}
//Ok im not even pretending to know what this does
unsafe impl Send for LlamaModel {}

#[pymethods]
impl LlamaModel {
    #[new]
    fn new(path: String, context_length: usize) -> Self {
        let model = Model::load(&path,context_length, |_| {}).unwrap();
        LlamaModel{model:model}
    }

    fn generate(&self, prompt: String, maximum_token_count:Option<usize>,py_callback: Option<PyObject>) ->  PyResult<()> {
        let session_params = InferenceSessionParameters::default();
        let geneartion_params = InferenceParameters::default();
        let mut session = self.model.start_session(session_params);
        let mut rng = rand::thread_rng();
        let prompt = prompt.as_str();

        fn callback(tokens:&str)-> Result<(), Error> {
            println!("{}",tokens);
            Ok(())
        }

        let result = session.inference_with_prompt(&self.model,
            &geneartion_params,
            &prompt,
            maximum_token_count,
            &mut rng,
            callback);
        Ok(())
    }
}


/// A Python module implemented in Rust.
#[pymodule]
fn llama_rs_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LlamaModel>()?;
    Ok(())
}