from typing import Optional, Callable
from enum import Enum
from .config import GenerationConfig, SessionConfig
from .results import GenerationResult

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