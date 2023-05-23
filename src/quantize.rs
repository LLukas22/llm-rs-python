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
        ContainerType::GGJT => ggml::format::SaveContainerType::GgjtV2,
    };

    let quantization = match quantization {
        QuantizationType::Q4_0 => Ok(ggml::Type::Q4_0),
        QuantizationType::Q4_1 => Ok(ggml::Type::Q4_1),
        QuantizationType::F16 => Err(QuantizeError::UnsupportedElementType {
            element_type: ggml::Type::F16,
        }),
    }?;

    let mut source = BufReader::new(std::fs::File::open(source)?);
    let mut destination = BufWriter::new(std::fs::File::create(destination)?);

    quantize::<M, _, _>(
        &mut source,
        &mut destination,
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
