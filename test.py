from llama_rs_python import LlamaModel

model = LlamaModel("../llama-rs/ggml-alpaca-7b-q4.bin",512)
model.generate("Rust is a ", 100)