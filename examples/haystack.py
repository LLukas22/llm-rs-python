from haystack.nodes import PromptNode, PromptModel
from llm_rs.haystack import RustformersInvocationLayer

model = PromptModel("rustformers/redpajama-7b-ggml",
                    max_length=1024,
                    invocation_layer_class=RustformersInvocationLayer,
                    model_kwargs={"model_file":"RedPajama-INCITE-7B-Instruct-q5_1-ggjt.bin"})

pn = PromptNode(
    model,
    max_length=1024,
    default_prompt_template="question-answering-with-document-scores",
)

pn("Write me a short story about a lama riding a crab.",stream=True)

