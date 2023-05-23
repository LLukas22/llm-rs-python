from typing import Optional, Callable, List, Union
from abc import ABC
import os

from .config import GenerationConfig, SessionConfig, ContainerType, QuantizationType
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

    @property
    def lora_paths(self)->Optional[List[str]]: ...

    def  __init__(self,
                  path:Union[str,os.PathLike],
                  session_config:SessionConfig=SessionConfig(),
                  lora_paths:Optional[List[Union[str,os.PathLike]]]=None,
                  verbose:bool=False) -> None: ...
    
    def generate(self,prompt:str,
                 generation_config:Optional[GenerationConfig]=None,
                 callback:Callable[[str],Optional[bool]]=None) -> GenerationResult: 
        """
        Generates text from a prompt.
        """ 
        ...
    
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

    @staticmethod
    def quantize(source:str,destination:str,quantization:QuantizationType=QuantizationType.Q4_0,container:ContainerType=ContainerType.GGJT)->None:
        """
        Quantizes the model.
        """
        ...