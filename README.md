# llm-rs-python: Python Bindings for Rust's llm Library

[![PyPI](https://img.shields.io/pypi/v/llm-rs)](https://pypi.org/project/llm-rs/)
[![PyPI - License](https://img.shields.io/pypi/l/llm-rs)](https://pypi.org/project/llm-rs/)
[![Downloads](https://static.pepy.tech/badge/llm-rs)](https://pepy.tech/project/llm-rs)

Welcome to `llm-rs`, an unofficial Python interface for the Rust-based [llm](https://github.com/rustformers/llm) library, made possible through [PyO3](https://github.com/PyO3/pyo3). Our package combines the convenience of Python with the performance of Rust to offer an efficient tool for your machine learning projects. 🐍❤️🦀

With `llm-rs`, you can operate a variety of Large Language Models (LLMs) including LLama and GPT-NeoX directly on your CPU or GPU. 

For a detailed overview of all the supported architectures, visit the [llm](https://github.com/rustformers/llm) project page. 

### Integrations:
* 🦜️🔗 [LangChain](https://github.com/hwchase17/langchain)
* 🌾🔱 [Haystack](https://github.com/deepset-ai/haystack)

## Installation

Simply install it via pip: `pip install llm-rs`

<details>
<summary>Installation with GPU Acceleration Support</summary>
<br>

`llm-rs` incorporates support for various GPU-accelerated backends to facilitate enhanced inference times. To enable GPU-acceleration the `use_gpu` parameter of your `SessionConfig` must be set to `True`. We distribute prebuilt binaries for the following operating systems and graphics APIs:

### MacOS (Using Metal)
For MacOS users, the Metal-supported version of `llm-rs` can be easily installed via pip:

`
pip install llm-rs-metal
`

### Windows/Linux (Using CUDA for Nvidia GPUs)
Due to the significant file size, CUDA-supported packages cannot be directly uploaded to `pip`. To install them, download the appropriate `*.whl` file from the latest [Release](https://github.com/LLukas22/llm-rs-python/releases/latest) and install it using pip as follows:

`
pip install [wheelname].whl
`

### Windows/Linux (Using OpenCL for All GPUs)

For universal GPU support on Windows and Linux, we offer an OpenCL-supported version. It can be installed via pip:

`
pip install llm-rs-opencl
`
</details>


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

### Streaming Text
Text can be yielded from a generator via the `stream` function:
```python 
from llm_rs import AutoModel, KnownModels

#load the model
model = AutoModel.from_pretrained("path/to/model.bin",model_type=KnownModels.Llama)

#generate
for token in model.stream("The meaning of life is"):
    print(token)
```

### Running GGML models from the Hugging Face Hub
GGML converted models can be directly downloaded and run from the hub.
```python 
from llm_rs import AutoModel

model = AutoModel.from_pretrained("rustformers/mpt-7b-ggml",model_file="mpt-7b-q4_0-ggjt.bin")
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
## 🦜️🔗 LangChain Usage
Utilizing `llm-rs-python` through langchain requires additional dependencies. You can install these using `pip install llm-rs[langchain]`. Once installed, you gain access to the `RustformersLLM` model through the `llm_rs.langchain` module. This particular model offers features for text generation and embeddings.

Consider the example below, demonstrating a straightforward LLMchain implementation with MPT-Instruct:

```python
from llm_rs.langchain import RustformersLLM
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template="""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:
Answer:"""

prompt = PromptTemplate(input_variables=["instruction"],template=template,)

llm = RustformersLLM(model_path_or_repo_id="rustformers/mpt-7b-ggml",model_file="mpt-7b-instruct-q5_1-ggjt.bin",callbacks=[StreamingStdOutCallbackHandler()])

chain = LLMChain(llm=llm, prompt=prompt)

chain.run("Write a short post congratulating rustformers on their new release of their langchain integration.")
```


## 🌾🔱 Haystack Usage
Utilizing `llm-rs-python` through haystack requires additional dependencies. You can install these using `pip install llm-rs[haystack]`. Once installed, you gain access to the `RustformersInvocationLayer` model through the `llm_rs.haystack` module. This particular model offers features for text generation.

Consider the example below, demonstrating a straightforward Haystack-Pipeline implementation with OpenLLama-3B:

```python
from haystack.nodes import PromptNode, PromptModel
from llm_rs.haystack import RustformersInvocationLayer

model = PromptModel("rustformers/open-llama-ggml",
                    max_length=1024,
                    invocation_layer_class=RustformersInvocationLayer,
                    model_kwargs={"model_file":"open_llama_3b-q5_1-ggjt.bin"})

pn = PromptNode(
    model,
    max_length=1024
)

pn("Write me a short story about a lama riding a crab.",stream=True)
```


## Documentation

For in-depth information on customizing the loading and generation processes, refer to our detailed [documentation](https://llukas22.github.io/llm-rs-python/).