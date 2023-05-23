from typing import Optional,List
from enum import Enum, auto

class ContainerType(Enum):
    GGML=auto(),
    GGJT=auto(),

class QuantizationType(Enum):
    Q4_0=auto(),
    Q4_1=auto(),
    F16=auto(),

class Precision(Enum):
    FP32=auto(),
    FP16=auto(),
    
class GenerationConfig():
    """
    Configuration parameters for the generation process.
    """
    top_k:int
    top_p:float
    temperature:float
    repetition_penalty:float
    repetition_penalty_last_n:int
    seed:int
    max_new_tokens:Optional[int]
    stop_words:Optional[List[str]]
    
    def  __init__(self,top_k:int=40,top_p:float=0.95,temperature:float=0.8,repetition_penalty:float=1.3,repetition_penalty_last_n:int=512,seed:int=42,max_new_tokens:Optional[int]=None,stop_words:Optional[List[str]]=None) -> None: ...
    
class SessionConfig():
    """
    Configure the generation session that will be used for all generations.
    """
    threads:int
    batch_size:int
    keys_memory_type:Precision
    values_memory_type:Precision
    
    
    @property
    def context_length(self)->int: ...
    
    @property
    def prefer_mmap(self)->bool:...
    
    def  __init__(self,threads:int=8,batch_size:int=8,context_length:int=2048,keys_memory_type:Precision=Precision.FP32,values_memory_type:Precision=values_memory_type.FP32,prefer_mmap:bool=True) -> None: ...