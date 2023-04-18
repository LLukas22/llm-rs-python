from llama_rs_python import LlamaModel,GenerationIterator
from typing import  Optional
import sys 

model = LlamaModel("../llama-rs/ggml-alpaca-7b-q4.bin",512)
print("Loaded model!")
print("Begining Generation:")
prompt="Human: Are you a good little KI?\nKI:"
print(prompt,end=" ")

counter = 0

def callback(string)->Optional[bool]:
    global counter
    print(string,end="")
    sys.stdout.flush()
    
model.generate(prompt, 512, callback)

# iterator = GenerationIterator(size=10_000)
# for i in iterator:
#     print(i)