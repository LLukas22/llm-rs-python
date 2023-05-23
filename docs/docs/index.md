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

### Model Loading and Deployment

This library presently utilizes the [ggml](https://github.com/ggerganov/ggml) backend, and necessitates ggml-converted models for inference. 

These converted models can be readily found on the [HuggingfaceHub](https://huggingface.co/models?search=ggml) or referenced in the "[Known-good-Models](https://github.com/rustformers/llm/blob/main/known-good-models.md)" list, curated in the `rustformers/llm` repository.

Each model architecture can be conveniently loaded through the `llm_rs` module.

For instance, loading an MPT model (like [mpt-7b](https://huggingface.co/LLukas22/mpt-7b-ggml)) can be achieved as shown below:

```python
from llm_rs import Mpt

model = Mpt("path/to/model.bin")
```

### Streamlined Model Loading 

Models that have been converted or quantized using `llm-rs` include an additional `*.meta` file. This file enables streamlined loading via the `AutoModel` module. This feature is particularly advantageous as it allows you to load models without specifying the underlying architecture.

The following is an illustrative example:

```python
from llm_rs import AutoModel

model = AutoModel.load("path/to/model.bin")
```

In this streamlined approach, the `AutoModel` module automatically infers the architecture from the `*.meta` file associated with the model, providing an intuitive and straightforward method for loading your models.

### Text Generation

Every model implements a `generate` function that you can use to generate text using the loaded model. 

Here's a quick look at how it works:

```python
result = model.generate("The meaning of life is")
print(result.text)
```

The `generate` function returns a result object, which contains the generated text. 

### A complete Example

Combining the previous examples, here's the simplest example of text generation:

```python
from llm_rs import Llama

model = Llama("path/to/model.bin")
result = model.generate("The meaning of life is")
print(result.text)
```

## Customize Model Loading  

### Custom Session Configuration

When loading a model, you can pass a `SessionConfig` to customize certain runtime parameters. Here are the parameters you can adjust:

| Parameter          | Description                                           | Default Value    |
| ------------------ | ----------------------------------------------------- | ---------------- |
| `threads`          | Defines the number of threads for the generation process. | `8`              |
| `batch_size`       | Specifies the size of the batch to be processed concurrently. | `8`              |
| `context_length`   | Sets the length of the context for generation.        | `2048`           |
| `keys_memory_type` | Selects the memory precision type for keys.          | `Precision.FP32` |
| `values_memory_type` | Chooses the memory precision type for values.       | `Precision.FP32` |
| `prefer_mmap`      | Determines if memory mapping is preferred.           | `True`           |

To illustrate, here's an example of loading a model with a custom `SessionConfig`:

```python
from llm_rs import Llama, SessionConfig, Precision

session_config = SessionConfig(
    threads=12,
    context_length=512,
    prefer_mmap=False,
    keys_memory_type=Precision.FP16,
    values_memory_type=Precision.FP16
)
model = Llama("path/to/model.bin", session_config=session_config)
```

In this example, we've configured the session to use 12 threads, a context length of 512, and disabled memory mapping. Both keys and values will be stored with a precision of FP16.


### Support for LoRA Adapters

[LoRA](https://arxiv.org/abs/2106.09685) adapters, a transformative method for reducing the memory footprint of transformer models, are compatible with all model architectures in `llm-rs`. Before use, LoRA adapters must be converted into the ggml format. These can then be loaded by passing them to the model's constructor as shown below:

```python
from llm_rs import Llama

model = Llama("path/to/model.bin", lora_paths=["path/to/lora.bin"])
```

#### Using multiple LoRA Adapters

If multiple [LoRA](https://arxiv.org/abs/2106.09685) adapters should be used they can simply be added by passing multiple files to the `lora_paths` parameter.

```python
from llm_rs import Llama

model = Llama("path/to/model.bin", lora_paths=["path/to/lora_1.bin","path/to/lora_2.bin"])
```

### Verbose Loading for Detailed Insights

For a more comprehensive understanding of the loading process, the `verbose` flag can be utilized. By setting `verbose` to `True`, the library will provide detailed output at each step of the loading process, offering valuable insights for debugging or optimization.

```python
from llm_rs import Llama

model = Llama("path/to/model.bin", verbose=True)
```

This enhanced verbosity aids in tracking the model loading procedure, ensuring smooth and efficient operations.

## Customize Generation

### Fine-Tuning the Generation Process

The generation method offers a significant degree of customizability through the `GenerationConfig` object. This configuration allows for precise control over the sampling of tokens during generation. Here are the parameters that you can adjust:

| Parameter                | Description                                                   | Default Value   |
| ------------------------ | ------------------------------------------------------------- | --------------- |
| `top_k`                  | The number of top tokens to be considered for generation.     | `40`            |
| `top_p`                  | The cumulative probability cutoff for token selection.        | `0.95`          |
| `temperature`            | The softmax temperature for controlling randomness.           | `0.8`           |
| `repetition_penalty`     | The penalty for token repetition.                             | `1.3`           |
| `repetition_penalty_last_n` | The penalty applied to the last N tokens if repeated.      | `512`           |
| `seed`                   | The random seed for generating deterministic results.         | `42`            |
| `max_new_tokens`         | The maximum number of new tokens to generate (optional).      | `None`          |
| `stop_words`             | A list of words to stop generation upon encountering (optional). | `None`        |

For instance, you can set up a custom generation configuration as follows:

```python
from llm_rs import Llama, GenerationConfig

model = Llama("path/to/model.bin")

generation_config = GenerationConfig(top_p=0.8, seed=69)
result = model.generate("The meaning of life is", generation_config=generation_config)
```

### Implementing Callbacks During Generation

To further enhance your control over the generation process, `llm-rs` provides the ability to register a callback for each token generated. This is achieved by passing a function to the `generate` method. This function should accept a `String` and optionally return a `Bool`. If the returned value is `True`, the generation process will be halted.

Here's an example of how to use a callback:

```python
from llm_rs import Llama
from typing import Optional

model = Llama("path/to/model.bin")

def callback(token: str) -> Optional[bool]:
    print(token, end="")

result = model.generate("The meaning of life is", callback=callback)
```

In this example, the callback function simply prints each generated token without halting the generation process.

## Leveraging Tokenization Features

The `llm-rs` package provides direct access to the tokenizer and vocabulary of the loaded models through the `tokenize` and `decode` functions. These functions offer a straightforward way to convert between plain text and the corresponding tokenized representation.

Here's how you can make use of these functionalities:

```python
from llm_rs import Llama

model = Llama("path/to/model.bin")

# Convert plain text to tokenized representation
tokenized_text = model.tokenize("The meaning of life is")

# Convert tokenized representation back to plain text
decoded_text = model.decode(tokenized_text)
```

In this example, the `tokenize` function transforms a given text into a sequence of tokens that the model can understand. The `decode` function does the inverse by converting the tokenized representation back into human-readable text.
