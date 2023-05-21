from .config import QuantizationType,ContainerType,SessionConfig
from pydantic import BaseModel
import os
import pathlib
from .models import Mpt,GptNeoX
from .base_model import Model
import logging
from typing import Optional, List, Union,Type
import os
from enum import Enum, auto
from dataclasses import dataclass
import json
from blake3 import blake3

class KnownModels(Enum):
    GptNeoX = auto()
    Mpt = auto()


_QUANTIZATION_TYPE_MAP = {
    "Q4_0": QuantizationType.Q4_0,
    "Q4_1": QuantizationType.Q4_1,
    "F16": QuantizationType.F16
}

_CONTAINER_TYPE_MAP = {
    "GGML": ContainerType.GGML,
    "GGJT": ContainerType.GGJT
}

@dataclass()
class ModelMetadata():
    """
    A dataclass to store metadata about a model.
    """
    model: KnownModels
    quantization: QuantizationType
    container: ContainerType
    converter:str
    hash:Optional[str]=None

    def add_hash(self,model_path:Union[str,os.PathLike]):
        h  = blake3(max_threads=blake3.AUTO)
        b  = bytearray(128_000_000)
        mv = memoryview(b)
        with open(model_path, 'rb', buffering=0) as f:
            for n in iter(lambda : f.readinto(mv), 0):
                h.update(mv[:n])
        self.hash=h.hexdigest()

    def serialize(self):
        return {
            "model": self.model.name,
            "quantization": repr(self.quantization).split(".")[-1],
            "container": repr(self.container).split(".")[-1],
            "converter": self.converter,
            "hash": self.hash
        }
    
    @staticmethod
    def deserialize(metadata_dict):
        return ModelMetadata(
            model = KnownModels[metadata_dict["model"]],
            quantization = _QUANTIZATION_TYPE_MAP[metadata_dict["quantization"]],
            container = _CONTAINER_TYPE_MAP[metadata_dict["container"]],
            converter = metadata_dict["converter"],
            hash = metadata_dict["hash"]
        )




class AutoModel():
    """
    Utility to load models, without having to specify the model type.
    """
    @staticmethod
    def infer_model_type(model_file:Union[str,os.PathLike])->Type[Model]:
        path = pathlib.Path(model_file)
        if not path.is_file():
            raise ValueError(f"Model file '{model_file}' is not a file!")
        if not path.exists():
            raise ValueError(f"Model file '{model_file}' does not exist!")

        metadata_file = path.with_suffix(".meta")
        if not metadata_file.exists():
            raise ValueError(f"Model file '{model_file}' does not have a metadata file '{metadata_file}'!")
        
        metadata = ModelMetadata.deserialize(json.loads(metadata_file.read_text()))
        if metadata.model == KnownModels.GptNeoX:
            return GptNeoX
        elif metadata.model == KnownModels.Mpt:
            return Mpt
        else:
            raise ValueError(f"Unknown model type '{metadata.model}'")
        
    @staticmethod
    def load(path:Union[str,os.PathLike],
                  session_config:SessionConfig=SessionConfig(),
                  lora_paths:Optional[List[Union[str,os.PathLike]]]=None,
                  verbose:bool=False)->Model:
        
        model_type = AutoModel.infer_model_type(path)
        return model_type(path,session_config,lora_paths,verbose)
            
            

    
