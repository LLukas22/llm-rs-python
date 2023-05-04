from typing import Optional, Callable, List
from abc import ABC, abstractmethod

from .config import GenerationConfig, SessionConfig
from .results import GenerationResult

#Theoretically this is incorrect as the 'model' doesnt actually exist, but it is a good enough approximation for now. 
class Model(ABC):
    """
    Wrapper around a llm model.
    """
    config:SessionConfig
    
    @property
    def path(self)->str: ...
    
    @property
    def verbose(self)->bool: ...
    
    def  __init__(self,path:str,session_config:SessionConfig=SessionConfig(), verbose:bool=False) -> None: ...
    
    def generate(self,prompt:str,
                 generation_config:Optional[GenerationConfig]=None,
                 callback:Callable[[str],Optional[bool]]=None) -> GenerationResult: ...
    
    def tokenize(self,text:str) -> List[int]:
        """
        Tokenizes a string into a list of tokens.
        """
        ...
    
    def decode(self,tokens:List[int]) -> str:
        """
        Decodes a list of tokens into a string.
        """
        ...
    

class Llama(Model):
    """
    Wrapper around all Llama based models.
    """
    ...
    
    
class GPTJ(Model):
    """
    Wrapper around all GPTJ based models.
    """
    ...
    
class GPT2(Model):
    """
    Wrapper around all GPT2 based models.
    """
    ...
    
class Bloom(Model):
    """
    Wrapper around all Bloom based models.
    """
    ...
    
class NeoX(Model):
    """
    Wrapper around all GPT-NeoX based models.
    """
    ...