# llm-rs-python: Python Bindings for Rust's llm Library

Welcome to `llm-rs`, an unofficial Python interface for the Rust-based [llm](https://github.com/rustformers/llm) library, made possible through [PyO3](https://github.com/PyO3/pyo3). Our package combines the convenience of Python with the performance of Rust to offer an efficient tool for your machine learning projects. 🐍❤️🦀

With `llm-rs`, you can operate a variety of Large Language Models (LLMs) including LLama and GPT-NeoX directly on your CPU. 

For a detailed overview of all the supported architectures, visit the [llm](https://github.com/rustformers/llm) project page. 

## Installation

Simply install it via pip: `pip install llm-rs`

## Usage
### Running local GGML models:
Models can be loaded via the `AutoModel` interface.

```python 
from llm_rs import AutoModel, KnownModels

#load the model
model = AutoModel.from_pretrained("path/to/model.bin",model_type=KnownModels.Llama)

#generate
print(model.generate("The meaning of life is"))
```
### Running GGML models from the Hugging Face Hub
GGML converted models can be directly downloaded and run from the hub.
```python 
from llm_rs import AutoModel

model = AutoModel.from_pretrained("LLukas22/mpt-7b-ggml",model_file="mpt-7b-q4_0-ggjt.bin")
```
If there are multiple models in a repo the `model_file` has to be specified.
If you want to load repositories which were not created throught this library, you have to specify the `model_type` parameter as the metadata files needed to infer the architecture are missing.

### Running Pytorch Transfomer models from the Hugging Face Hub
`llm-rs` supports automatic conversion of all supported transformer architectures on the Huggingface Hub. 

To run covnersions additional dependencies are needed which can be installed via `pip install llm-rs[convert]`.

The models can then be loaded and automatically converted via the `from_pretrained` function.

```python
from llm_rs import AutoModel

model = AutoModel.from_pretrained("mosaicml/mpt-7b")
```

### Convert Huggingface Hub Models

The following example shows how a [Pythia](https://huggingface.co/EleutherAI/pythia-410m) model can be covnverted, quantized and run.

```python
from llm_rs.convert import AutoConverter
from llm_rs import AutoModel, AutoQuantizer
import sys

#define the model which should be converted and an output directory
export_directory = "path/to/directory" 
base_model = "EleutherAI/pythia-410m"

#convert the model
converted_model = AutoConverter.convert(base_model, export_directory)

#quantize the model (this step is optional)
quantized_model = AutoQuantizer.quantize(converted_model)

#load the quantized model
model = AutoModel.load(quantized_model,verbose=True)

#generate text
def callback(text):
    print(text,end="")
    sys.stdout.flush()

model.generate("The meaning of life is",callback=callback)
```

## Documentation

For in-depth information on customizing the loading and generation processes, refer to our detailed [documentation](https://llukas22.github.io/llm-rs-python/).