# llm-rs-python
Unofficial python bindings for the rust [llm](https://github.com/rustformers/llm) library created with [PyO3](https://github.com/PyO3/pyo3). 🐍❤️🦀

This package allows you to run multiple different Large Language Models (LLMs) like LLama or GPT-NeoX on your CPU.

All supported architectures are listed on the [llm](https://github.com/rustformers/llm) project page.


## Installation

Simply install it via pip: `pip install llm-rs`

## Usage

The package is type-hinted for easy usage.

A llama model can be run like this:

```python 
from llm_rs import Llama

#load the model
model = Llama("path/to/model.bin")

#generate
print(model.generate("The meaning of life is"))
```

The package also supports callbacks to get each token as it is generated.
The callback-function also supports canceling the generation by returning a `True` value from the pytohn side.

```python 
from llm_rs import Llama
import sys
from typing import Optional

#load the model
model = Llama("path/to/model.bin")

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
from llm_rs import Llama, GenerationConfig

#load the model
model = Llama("path/to/model.bin")

#create a config
config = GenerationConfig(top_p=0.9,seed=1441,max_new_tokens=1024)

#generate
print(model.generate("The meaning of life is",generation_config=config))
```

To configure model specific settings the `SessionConfig` class can be used.

```python
from llm_rs import Llama, SessionConfig

#define the session
session_config = SessionConfig(threads=8,context_length=512)

#load the model
model = Llama("path/to/model.bin",session_config=session_config)
```