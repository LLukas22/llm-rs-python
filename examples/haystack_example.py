from haystack.nodes import PromptModel
from llm_rs.haystack import RustformersInvocationLayer
from llm_rs import KnownModels,SessionConfig


#Enable `use_gpu` to use GPU acceleration
session_config = SessionConfig(use_gpu=False)
    
model = PromptModel("TheBloke/Llama-2-7B-Chat-GGML",
                    max_length=4096,
                    invocation_layer_class=RustformersInvocationLayer,
                    model_kwargs={
                        "model_file":"llama-2-7b-chat.ggmlv3.q4_K_S.bin",
                        "session_config":session_config,
                        "verbose":True,
                        })

prompt= """
System: You are a helpful, respectful and honest assistant.
User: Tell me a Story about a Lama riding the crab named Ferris in about 1000 words.
Assistant:
"""
model.invoke(prompt=prompt,stream=True)