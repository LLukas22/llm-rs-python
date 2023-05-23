# Model Conversion and Quantization

To employ transformers/pytorch models within `llm-rs`, it is essential to convert them into the GGML model format. Originally, this conversion process is facilitated through scripts provided by the original implementations of the models. An example can be found [here](https://github.com/ggerganov/ggml/blob/master/examples/mpt/convert-h5-to-ggml.py). The scripts will generate a GGML model in an `fp16` format, which can be utilized with `llm-rs`. However, for optimal performance and efficient usage, it is advisable to proceed with quantizing the models.

## Automated Conversion Capabilities

The `llm-rs` package offers a powerful feature to automatically convert any supported Huggingface transformers model to a format compatible with our system. This ease-of-use functionality requires additional dependencies which can be installed using the command: `pip install llm-rs[convert]`.

You can access the automatic conversion feature via the `llm_rs.convert` module. Within this module, an `AutoConverter` utility is provided. This intelligent tool automatically discerns the architecture of a specified Huggingface model and selects the appropriate converter for the job.

Here is an illustrative example of the `AutoConverter` in action:

```python
from llm_rs.convert import AutoConverter

# Specify the model to be converted and an output directory
export_folder = "path/to/folder" 
base_model = "EleutherAI/pythia-410m"

# Perform the model conversion
converted_model = AutoConverter.convert(base_model, export_folder)
```

If you provide a directory as an export target, the `AutoConverter` will automatically generate the name for the converted model. If you provide a specific file path, this path will be directly used for the converted model. The conversion function then returns the file path of the converted model.

## Manual Conversion

In cases where the `AutoConverter` is unable to accurately determine the architecture of your model, you have the option to manually specify the architecture by selecting the corresponding converter. These specialized converters are found in the `llm_rs.convert.models` module.

Here is an example of manual conversion:

```python
from llm_rs.convert.models import Gpt2Converter

# Define the model to be converted and the output path
base_model = "cerebras/Cerebras-GPT-111M"
export_file = "path/to/model.bin"

# Perform the model conversion
Gpt2Converter(base_model).convert(export_file)
```

In this example, the `Gpt2Converter` is specifically chosen to convert the specified model and save it to the provided file path. With this feature, you retain full control over the conversion process, ensuring the best compatibility with `llm-rs`.

## Optimize with Model Quantization

Quantization is an optimization strategy that can dramatically enhance the computational efficiency and minimize the memory footprint of your models. `llm-rs` offers tools for automated and manual quantization, adapting models to perform more efficiently.

### Auto Quantization
Models converted via `llm-rs` are readily compatible with our `AutoQuantizer`. The process of auto quantization is straightforward, as demonstrated below:

```python
from llm_rs import AutoQuantizer

# Define the path of the converted model
converted_model="path/to/model.bin"

# Quantize the model
quantized_model = AutoQuantizer.quantize(converted_model)
```

### Manual Quantization

`llm-rs` provides a `quantize` function specific to each supported architecture to automate the quantization process. The function requires the input of the converted GGML model and the specified quantization format. An example is as follows:

```python
from llm_rs import Mpt, QuantizationType, ContainerType

Mpt.quantize("path/to/source.bin",
    "path/to/destination.bin",
    quantization=QuantizationType.Q4_0,
    container=ContainerType.GGJT
)
```

In this instance, the `quantize` function refines the original GGML model into a more efficient quantized version, optimizing it for execution on your chosen hardware. This step not only boosts performance but also yields significant savings in storage space.

### Deploying Your Quantized Model

Upon quantizing your GGML model, you can immediately deploy it. Here's a simple demonstration of loading the quantized model and initiating a text generation:

```python
from llm_rs import Mpt

# Load the quantized model
model = Mpt("path/to/destination.bin")

# Initiate a text generation
result = model.generate("The meaning of life is")

# Display the generated text
print(result.text)
```

This example highlights how straightforward it is to execute a text generation with your quantized model, enabling you to reap the benefits of enhanced computational efficiency and minimized memory footprint.
