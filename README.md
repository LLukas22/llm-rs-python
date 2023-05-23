# llm-rs-python: Python Bindings for Rust's llm Library

Welcome to `llm-rs`, an unofficial Python interface for the Rust-based [llm](https://github.com/rustformers/llm) library, made possible through [PyO3](https://github.com/PyO3/pyo3). Our package combines the convenience of Python with the performance of Rust to offer an efficient tool for your machine learning projects. 🐍❤️🦀

With `llm-rs`, you can operate a variety of Large Language Models (LLMs) including LLama and GPT-NeoX directly on your CPU. 

For a detailed overview of all the supported architectures, visit the [llm](https://github.com/rustformers/llm) project page. 

## Installation

Simply install it via pip: `pip install llm-rs`

## Usage
### Running GGML converted models:
This example shows how a Llama model can be loaded.

```python 
from llm_rs import Llama

#load the model
model = Llama("path/to/model.bin")

#generate
print(model.generate("The meaning of life is"))
```

### Running Huggingface Hub Models
`llm-rs` supports automatic conversion of all supported transformer architectures on the Huggingface Hub. 

To run covnersions additional dependencies are needed which can be installed via `pip install llm-rs[convert]`.

The following example shows how a [Pythia](https://huggingface.co/EleutherAI/pythia-410m) model can be covnverted, quantized and run.

```python
from llm_rs.convert import AutoConverter
from llm_rs import AutoModel, AutoQuantizer
import sys

#define the model which should be converted and an output folder
export_folder = "path/to/folder" 
base_model = "EleutherAI/pythia-410m"

#convert the model
converted_model = AutoConverter.convert(base_model, export_folder)

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