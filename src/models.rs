use pyo3::prelude::*;

use crate::model_base::{wrap_model};

wrap_model!(Llama,llm::models::Llama);

wrap_model!(GPTJ,llm::models::GptJ);

wrap_model!(GPT2,llm::models::Gpt2);

wrap_model!(NeoX,llm::models::NeoX);

wrap_model!(Bloom,llm::models::Bloom);