use llm_base::QuantizeProgress;
use pyo3::prelude::*;

use std::{
    io::{BufReader, BufWriter},
    path::PathBuf,
};

use llm::{quantize, QuantizeError};

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum ContainerType {
    GGML,
    GGJT,
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantizationType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    F16,
}

pub fn _quantize<M: llm::KnownModel + 'static>(
    source: PathBuf,
    destination: PathBuf,
    container: ContainerType,
    quantization: QuantizationType,
    progress_callback: impl Fn(String),
) -> Result<(), QuantizeError> {
    let container = match container {
        ContainerType::GGML => llm_base::ggml::format::SaveContainerType::Ggml,
        ContainerType::GGJT => llm_base::ggml::format::SaveContainerType::GgjtV3,
    };

    let quantization = match quantization {
        QuantizationType::Q4_0 => Ok(llm_base::ggml::Type::Q4_0),
        QuantizationType::Q4_1 => Ok(llm_base::ggml::Type::Q4_1),
        QuantizationType::Q5_0 => Ok(llm_base::ggml::Type::Q5_0),
        QuantizationType::Q5_1 => Ok(llm_base::ggml::Type::Q5_1),
        QuantizationType::Q8_0 => Ok(llm_base::ggml::Type::Q8_0),
        QuantizationType::F16 => Err(QuantizeError::UnsupportedElementType {
            element_type: llm_base::ggml::Type::F16,
        }),
    }?;

    let mut source_reader = BufReader::new(std::fs::File::open(&source)?);
    let mut destination_reader = BufWriter::new(std::fs::File::create(destination)?);
    let vocabulary = llm::TokenizerSource::Embedded.retrieve(&source).unwrap();

    quantize::<M, _, _>(
        &mut source_reader,
        &mut destination_reader,
        vocabulary,
        container,
        quantization,
        |progress| match progress {
            QuantizeProgress::HyperparametersLoaded => {
                progress_callback("Loaded hyperparameters".to_string())
            }
            QuantizeProgress::TensorLoading {
                name,
                dims,
                element_type,
                n_elements,
            } => progress_callback(format!(
                "Loading tensor `{name}` ({n_elements} ({dims:?}) {element_type} elements)"
            )),
            QuantizeProgress::TensorQuantizing { name } => {
                progress_callback(format!("Quantizing tensor `{name}`"))
            }
            QuantizeProgress::TensorQuantized {
                name,
                original_size,
                reduced_size,
                history,
            } => progress_callback(format!(
        "Quantized tensor `{name}` from {original_size} to {reduced_size} bytes ({history:?})"
    )),
            QuantizeProgress::TensorSkipped { name, size } => {
                progress_callback(format!("Skipped tensor `{name}` ({size} bytes)"))
            }
            QuantizeProgress::Finished {
                original_size,
                reduced_size,
                history,
            } => progress_callback(format!(
                "Finished quantization from {original_size} to {reduced_size} bytes ({history:?})"
            )),
        },
    )
}
