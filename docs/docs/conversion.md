# Model Conversion and Quantization

To employ transformers/pytorch models within `llm-rs`, it is essential to convert them into the GGML model format. As of now, this conversion process is facilitated through scripts provided by the original implementations of the models. An example can be found [here](https://github.com/ggerganov/ggml/blob/master/examples/mpt/convert-h5-to-ggml.py). The scripts will generate a GGML model in an `fp16` format, which can be utilized with `llm-rs`. However, for optimal performance and efficient usage, it is advisable to proceed with quantizing the models.

## Efficient Models through Quantization

Quantization is a powerful technique that can significantly improve the computational efficiency and reduce the memory footprint of your models. With `llm-rs`, we offer a `quantize` function for every architecture, designed to automate this process for you.

This process requires you to provide the converted GGML model and define the quantization format. The following is an illustrative example:

```python
from llm_rs import Mpt, QuantizationType, ContainerType

Mpt.quantize("path/to/source.bin",
    "path/to/destination.bin",
    quantization=QuantizationType.Q4_0,
    container=ContainerType.GGJT
)
```

In this example, the `quantize` function transforms the original GGML model into a quantized version, which is more suited for efficient execution on your hardware. This step not only enhances performance but can also result in substantial storage savings.

### Running Your Quantized Model

Once you have quantized your GGML model, you can immediately put it to use. Here is a basic example of loading the model and running a generation:

```python
from llm_rs import Mpt

# Load the quantized model
model = Mpt("path/to/destination.bin")

# Generate text
result = model.generate("The meaning of life is")

# Output the generated text
print(result.text)
```
