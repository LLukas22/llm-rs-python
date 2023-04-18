# llama-rs-python
Unofficial python bindings for [llama-rs](https://github.com/rustformers/llama-rs) created with [PyO3](https://github.com/PyO3/pyo3). 🐍❤️🦀

This package gives access to the basic functionality of the llama-rs project.

GGML converted models can be loaded and executed.

## Installation

Simply install it via pip: `pip install llama-rs-python`

## Usage

The package is typehinted for easy usage.

A usage example could look like this:

```python 
from llama_rs_python import Model

#load the model
model = Model("path/to/model.bin")

#generate
print(model.generate("The meaning of life is").text)
```

The package also supports callbacks to get each token as it is generated.
The callback-function also supports canceling the generation by returning a `True` value from the pytohn side.

```python 
import sys
from llama_rs_python import Model

#load the model
model = Model("path/to/model.bin")

#define the callback
def callback(token:str)->Optional[bool]:
    print(token,end="")
    sys.stdout.flush()
    # (return True here to cancel the generation)

#start generation
model.generate("The meaning of life is",callback=callback)
```

The configuration of the generation is handled by the `GenerationConfig` class.

```python 
from llama_rs_python import Model, GenerationConfig

#load the model
model = Model("path/to/model.bin")

#create a config
config = GenerationConfig(top_p=0.9)

#generate
print(model.generate("The meaning of life is",generation_config=config).text)
```

