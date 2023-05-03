from llm_rs import GPTJ,SessionConfig,Precision,GenerationConfig
from llm_rs.results import GenerationResult
from typing import Optional
import sys 

prompt="The meaning of life is"

llama_model = "ggjt-model.bin"
gpt_j_model = "gpt-j-ggml-model-q4_0.bin"
print("Loading Model...")
session_config = SessionConfig(threads=8,context_length=512,prefer_mmap=True)
model = GPTJ(gpt_j_model,session_config=session_config,verbose=True)

print("Loaded model!")

print("Begining Generation:")
print(prompt,end="")

def callback(token:str)->Optional[bool]:
    print(token,end="")
    sys.stdout.flush()
    
config=GenerationConfig(top_p=0.9,seed=42,max_new_tokens=1024)
result = model.generate(prompt,generation_config=config,callback=callback)
print(result)