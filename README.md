# llm-rs-python: Python Bindings for Rust's llm Library

Welcome to `llm-rs`, an unofficial Python interface for the Rust-based [llm](https://github.com/rustformers/llm) library, made possible through [PyO3](https://github.com/PyO3/pyo3). Our package combines the convenience of Python with the performance of Rust to offer an efficient tool for your machine learning projects. 🐍❤️🦀

With `llm-rs`, you can operate a variety of Large Language Models (LLMs) including LLama and GPT-NeoX directly on your CPU. 

For a detailed overview of all the supported architectures, visit the [llm](https://github.com/rustformers/llm) project page. 

## Installation

Simply install it via pip: `pip install llm-rs`

## Usage

The package is type-hinted for easy usage.

A Llama model can be run like this:

```python 
from llm_rs import Llama

#load the model
model = Llama("path/to/model.bin")

#generate
print(model.generate("The meaning of life is"))
```

## Documentation

For in-depth information on customizing the loading and generation processes, refer to our detailed [documentation](https://llukas22.github.io/llm-rs-python/).