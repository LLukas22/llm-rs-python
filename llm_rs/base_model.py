from typing import Optional, Callable, List, Union, Generator
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
                  tokenizer_name_or_path:Optional[Union[str,os.PathLike]]=None,
                  lora_paths:Optional[List[Union[str,os.PathLike]]]=None,
                  verbose:bool=False) -> None: ...
    
    def generate(self,prompt:str,
                 generation_config:Optional[GenerationConfig]=None,
                 callback:Optional[Callable[[str],Optional[bool]]]=None) -> GenerationResult: 
        """
        Generates text from a prompt.
        """ 
        ...

    def generate(self,prompt:str) -> List[float]: 
        """
        Embed a given prompt into vector representation.
        """ 
        ...

    def stream(self,prompt:str,
                 generation_config:Optional[GenerationConfig]=None,
                 ) -> Generator[str,None,None]: 
        """
        Streams text from a prompt.
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
    def quantize(source:str,destination:str,quantization:QuantizationType=QuantizationType.Q4_0,container:ContainerType=ContainerType.GGJT,callback:Optional[Callable[[str],None]]=None)->None:
        """
        Quantizes the model.
        """
        ...