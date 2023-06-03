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
) -> Result<(), QuantizeError> {
    let container = match container {
        ContainerType::GGML => ggml::format::SaveContainerType::Ggml,
        ContainerType::GGJT => ggml::format::SaveContainerType::GgjtV3,
    };

    let quantization = match quantization {
        QuantizationType::Q4_0 => Ok(ggml::Type::Q4_0),
        QuantizationType::Q4_1 => Ok(ggml::Type::Q4_1),
        QuantizationType::Q5_0 => Ok(ggml::Type::Q5_0),
        QuantizationType::Q5_1 => Ok(ggml::Type::Q5_1),
        QuantizationType::Q8_0 => Ok(ggml::Type::Q8_0),
        QuantizationType::F16 => Err(QuantizeError::UnsupportedElementType {
            element_type: ggml::Type::F16,
        }),
    }?;

    let mut source_reader = BufReader::new(std::fs::File::open(&source)?);
    let mut destination_reader = BufWriter::new(std::fs::File::create(destination)?);
    let vocabulary = llm::VocabularySource::Model.retrieve(&source).unwrap();

    
    quantize::<M, _, _>(
        &mut source_reader,
        &mut destination_reader,
        vocabulary,
        container,
        quantization,
        |progress| match progress {
            QuantizeProgress::HyperparametersLoaded => log::info!("Loaded hyperparameters"),
            QuantizeProgress::TensorLoading {
                name,
                dims,
                element_type,
                n_elements,
            } => println!(
                "Loading tensor `{name}` ({n_elements} ({dims:?}) {element_type} elements)"
            ),
            QuantizeProgress::TensorQuantizing { name } => log::info!("Quantizing tensor `{name}`"),
            QuantizeProgress::TensorQuantized {
                name,
                original_size,
                reduced_size,
                history,
            } => println!(
        "Quantized tensor `{name}` from {original_size} to {reduced_size} bytes ({history:?})"
    ),
            QuantizeProgress::TensorSkipped { name, size } => {
                println!("Skipped tensor `{name}` ({size} bytes)")
            }
            QuantizeProgress::Finished {
                original_size,
                reduced_size,
                history,
            } => println!(
                "Finished quantization from {original_size} to {reduced_size} bytes ({history:?})"
            ),
        },
    )
}
