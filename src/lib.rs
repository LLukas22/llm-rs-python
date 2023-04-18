use pyo3::prelude::*;

use pyo3::exceptions::PyException;
use pyo3::types::{PyTuple,PyBool, PyFunction};
use llama_rs::{Model, InferenceSessionParameters,InferenceParameters, TokenUtf8Buffer, InferenceError};
use std::io::{self, Write};

use rand::{prelude::*, Error};

#[pyclass]
pub struct GenerationStats(llama_rs::InferenceStats){

}

#[pyclass]
pub struct LlamaModel{
    model: Model
}
//Ok im not even pretending to know what this does
unsafe impl Send for LlamaModel {}

#[pyclass]
pub struct GenerationIterator{
    start: usize,
    end: usize,
    //iter: Box<dyn iter::Iterator<Item = usize> + Send>
}

#[pymethods]
impl GenerationIterator {
    #[new]
    fn new(size:usize) -> Self {
        GenerationIterator{start:0,end:size}
    }

    fn __iter__(slf: PyRef<'_, Self>) ->  PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) ->Option<usize> {
        if slf.start < slf.end{
            let temp = slf.start;
            slf.start += 1;
            return Some(temp);
        }else{
            return None;
        }
    }
}


#[pymethods]
impl LlamaModel {
    #[new]
    fn new(path: String, context_length: usize) -> Self {
        let model = Model::load(&path,context_length, |_| {}).unwrap();
        LlamaModel{model:model}
    }

    fn generate(&self,_py: Python,prompt: String
        ,maximum_token_count:Option<usize>,
        callback:Option<PyObject>) ->  PyResult<()> {

        let session_params = InferenceSessionParameters::default();
        let geneartion_params = InferenceParameters::default();
        let mut session = self.model.start_session(session_params);
        let mut rng = rand::thread_rng();
        let prompt = prompt.as_str();

        let mut token_utf8_buf = TokenUtf8Buffer::new();

        let mut callback_function:Option<&PyAny> = None;
        let pytohn_object: Py<PyAny>;
        if let Some(unwrapped) = callback{
            pytohn_object = unwrapped;

            callback_function = Some(pytohn_object.as_ref(_py));
        }

       session.feed_prompt(&self.model,
                &geneartion_params,
                &prompt,
                |_|{return Ok::<(), Error>(())});
        
       

        let mut tokens_processed = 0;    
        
        loop {
            if maximum_token_count.is_some() && tokens_processed >= maximum_token_count.unwrap() {
                break;
            }

            let token = match session.infer_next_token(&self.model, &geneartion_params, &mut rng) {
                Ok(token) => token,
                Err(InferenceError::EndOfText) => break,
                Err(e) => return Err(PyException::new_err(e.to_string())),
            };

            if let Some(tokens) = token_utf8_buf.push(token) {
                if let Some(function) = callback_function{
                    let args = PyTuple::new(_py, &[tokens]);
                    let result = function.call1(args)?;
                    if !result.is_none() {
                        if result.is_true()?{
                            break;
                        }
                    }
                }
            }
                

            tokens_processed += 1;
        }
        Ok(())
    }
}


/// A Python module implemented in Rust.
#[pymodule]
fn llama_rs_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LlamaModel>()?;
    m.add_class::<GenerationIterator>()?;
    Ok(())
}