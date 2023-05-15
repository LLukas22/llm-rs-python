# Introduction
`llm-rs` serves as an unofficial Python binding for the Rust [llm](https://github.com/rustformers/llm) crate, constructed using the [PyO3](https://github.com/PyO3/pyo3) library. This package combines the ease of Python with the efficiency of Rust, enabling the execution of multiple large language models (LLMs) on local hardware infrastructure with a minimal set of dependencies.

## Getting started

### How to Install

#### The pip Way
We have precompiled binaries for most platforms up for grabs on [PyPI](https://pypi.org/project/llm-rs/). Install them using pip with this straightforward command:

```shell
pip install llm-rs
```

#### The Local Build Way
If you're looking for local or development builds, you'll want to use [maturin](https://github.com/PyO3/maturin) as a build tool. Ensure it's installed with:

```shell
pip install maturin
```

Next, download and install the repo into your local Python environments using these commands:

```shell
git clone https://github.com/LLukas22/llm-rs-python
cd ./llm-rs-python
maturin develop -r
```

### Model Loading

Right now, this library leverages [ggml](https://github.com/ggerganov/ggml) as a backend and requires ggml-converted models to perform inference. 

You can find these models on the [HuggingfaceHub](https://huggingface.co/models?search=ggml) or in the "[Know-good-Models](https://github.com/rustformers/llm/blob/main/known-good-models.md)" list of the `rustformers/llm` repo.

All available model architectures can be accessed through the `llm_rs` module.

Here's a simple illustration of how to load a LLaMA model (like [gpt4all](https://huggingface.co/LLukas22/gpt4all-lora-quantized-ggjt)):

```python
from llm_rs import Llama
model = Llama("path/to/model.bin")
```

### Text Generation

Every model has a `generate` function that you can use to generate text using the loaded model. 

Here's a quick look at how it works:

```python
result = model.generate("The meaning of life is")
print(result.text)
```

The `generate` function returns a result object, which contains the generated text. 

### A Full Example

Combining the previous examples, here's the simplest example of text generation:

```python
from llm_rs import Llama
model = Llama("path/to/model.bin")
result = model.generate("The meaning of life is")
print(result.text)
```
