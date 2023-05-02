from typing import Optional,Callable
from enum import Enum

class GenerationConfig():
    """
    Configuration parameters for the generation process.
    """
    top_k:int
    top_p:float
    temperature:float
    repetition_penalty:float
    seed:int
    max_new_tokens:Optional[int]
    
    def  __init__(self,top_k:int=40,top_p:float=0.95,temperature:float=0.8,repetition_penalty:float=1.3,seed:int=42,max_new_tokens:Optional[int]=None) -> None: ...
    
    
class Precision(Enum):
    FP32=0,
    FP16=1,

class SessionConfig():
    """
    Configure the generation session that will be used for all generations.
    """
    threads:int
    batch_size:int
    repetition_penalty_last_n:int
    keys_memory_type:Precision
    values_memory_type:Precision
    
    @property
    def context_length(self)->int: ...
    
    def  __init__(self,threads:int=8,batch_size:int=8,context_length:int=2048,repetition_penalty_last_n:int=512,keys_memory_type:Precision=Precision.FP32,values_memory_type:Precision=values_memory_type.FP32) -> None: ...
    
class StopReason(Enum):
    """
    The reason why generation was stopped.
    """
    EndToken=0,
    MaxLength=1,
    UserCancelled=2

class GenerationTimes():
    """
    Contains the time taken for each step of generation in milliseconds.
    """
    @property
    def total(self)->int: ...
    @property
    def generation(self)->int: ...
    @property
    def prompt_feeding(self)->int: ...
    
class GenerationResult():
    """
    Containes the result of a generation and the time taken for each step.
    """
    @property
    def text(self)->str: ...
    @property
    def times(self)->GenerationTimes: ...
    @property
    def stop_reason(self)->StopReason: ...
    
    
class Model():
    """
    Wrapper around a ggml model.
    """
    config:SessionConfig
    verbose:bool
    
    def  __init__(self,path:str,session_config:SessionConfig=SessionConfig(), verbose:bool=False) -> None: ...
    
    def generate(self,prompt:str,
                 generation_config:Optional[GenerationConfig]=None,
                 callback:Callable[[str],Optional[bool]]=None) -> GenerationResult: ...