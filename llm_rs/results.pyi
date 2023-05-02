from enum import Enum

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