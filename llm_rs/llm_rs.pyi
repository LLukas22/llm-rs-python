from typing import Optional, Callable
from abc import ABC, abstractmethod

from .config import GenerationConfig, SessionConfig
from .results import GenerationResult

class InternalModel():
    config:SessionConfig
    
    @property
    def path(self)->str: ...
    
    @property
    def verbose(self)->bool: ...
    
    


#Theoretically this is incorrect as the 'model' doesnt actually exist, but it is a good enough approximation for now. 
class Model(ABC):
    """
    Wrapper around a llm model.
    """
    
    def  __init__(self,path:str,session_config:SessionConfig=SessionConfig(), verbose:bool=False) -> None: ...
    
    def generate(self,prompt:str,
                 generation_config:Optional[GenerationConfig]=None,
                 callback:Callable[[str],Optional[bool]]=None) -> GenerationResult: ...
    

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