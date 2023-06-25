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

pn("Life is",stream=True)

