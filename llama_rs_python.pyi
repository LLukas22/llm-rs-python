from  typing import Optional,Callable


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
    
class GenerationConfig():
    """
    Configuration parameters for the generation process.
    """
    top_k:int
    top_p:float
    temperature:float
    repetition_penalty:float
    def  __init__(self,top_k:int=40,top_p:float=0.95,temperature:float=0.8,repetition_penalty:float=1.3) -> None: ...
    

class Model():
    """
    Wrapper around a ggml model.
    """
    batch_size:int
    threads:int
    verbose:bool
    
    def  __init__(self,path:str,context_length:int=2048, threads:int=8, batch_size:int=8, verbose:bool=False) -> None: ...
    
    def generate(self,prompt:str,
                 generation_config:Optional[GenerationConfig]=None,
                 seed:Optional[int]=None,
                 maximum_token_count:Optional[int]=None,
                 callback:Callable[[str],Optional[bool]]=None) -> GenerationResult: ...