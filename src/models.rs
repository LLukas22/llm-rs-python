use pyo3::prelude::*;

use llm::models::{Llama as llm_Llama, GptJ as llm_GptJ};

use crate::configs;
use crate::results;
use crate::model_base::{self,Inferenze,_load};


#[pyclass]
pub struct Llama{
    internal_model: model_base::InternalModel
}

impl model_base::Inferenze for Llama{}

#[pymethods]
impl Llama {
    
    #[new]
    fn new(path: String, session_config:Option<configs::SessionConfig>,verbose:Option<bool>) -> Self {
        let model = _load::<llm_Llama>(path,session_config,verbose).unwrap();
        Llama { internal_model: model }
    } 

    fn generate(
        &self,
        _py: Python,
        prompt: String,
        generation_config: Option<&configs::GenerationConfig>,
        callback: Option<PyObject>,
    ) -> PyResult<results::GenerationResult> {

        return self._generate(_py,self.internal_model.llm_model.as_ref(),&self.internal_model.config,prompt, generation_config, callback);
    }
}

#[pyclass]
pub struct GPTJ{
    internal_model: model_base::InternalModel
}

impl model_base::Inferenze for GPTJ{}

#[pymethods]
impl GPTJ {
    
    #[new]
    fn new(path: String, session_config:Option<configs::SessionConfig>,verbose:Option<bool>) -> Self {
        let model = _load::<llm_GptJ>(path,session_config,verbose).unwrap();
        GPTJ { internal_model: model }
    } 

    fn generate(
        &self,
        _py: Python,
        prompt: String,
        generation_config: Option<&configs::GenerationConfig>,
        callback: Option<PyObject>,
    ) -> PyResult<results::GenerationResult> {

        return self._generate(_py,self.internal_model.llm_model.as_ref(),&self.internal_model.config,prompt, generation_config, callback);
    }
}