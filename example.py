from llm_rs import Model,SessionConfig,Precision,GenerationConfig

from typing import Optional
import sys 

prompt="The meaning of life is"

print("Loading Model...")
session_config = SessionConfig(threads=8,context_length=512)
model = Model("ggjt-model.bin",session_config=session_config,verbose=True)

print("Loaded model!")

print("Begining Generation:")
print(prompt,end="")

def callback(token:str)->Optional[bool]:
    print(token,end="")
    sys.stdout.flush()
    
config=GenerationConfig(top_p=0.9,seed=42,max_new_tokens=1024)
result = model.generate(prompt,generation_config=config,callback=callback)
print(result)