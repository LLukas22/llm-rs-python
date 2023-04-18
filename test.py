from llama_rs_python import Model,GenerationConfig
from typing import Optional
import sys 

prompt="The meaning of life is"

print("Loading Model...")
model = Model("../llama-rs/ggml-alpaca-7b-q4.bin",context_length=512,threads=8,batch_size=8,verbose=True)

print("Loaded model!")

print("Begining Generation:")
print(prompt,end="")

def callback(token:str)->Optional[bool]:
    print(token,end="")
    sys.stdout.flush()
    
config=GenerationConfig()
result = model.generate(prompt,generation_config=config,callback=callback)
print(result)