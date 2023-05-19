use pyo3::prelude::*;

use crate::model_base::wrap_model;

wrap_model!(Llama, llm::models::Llama);

wrap_model!(GptJ, llm::models::GptJ);

wrap_model!(Gpt2, llm::models::Gpt2);

wrap_model!(GptNeoX, llm::models::GptNeoX);

wrap_model!(Bloom, llm::models::Bloom);

wrap_model!(Mpt, llm::models::Mpt);
