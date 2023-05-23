from .config import QuantizationType,ContainerType,SessionConfig
from pydantic import BaseModel
import os
import pathlib
from .models import Mpt,GptNeoX,GptJ,Gpt2,Bloom,Llama
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
    GptJ = auto()
    Gpt2 = auto()
    Bloom = auto()
    Llama = auto()


_QUANTIZATION_TYPE_MAP = {
    "Q4_0": QuantizationType.Q4_0,
    "Q4_1": QuantizationType.Q4_1,
    "F16": QuantizationType.F16
}

_CONTAINER_TYPE_MAP = {
    "GGML": ContainerType.GGML,
    "GGJT": ContainerType.GGJT
}

_KNOWN_MODELS_MAP = {
    KnownModels.GptNeoX: GptNeoX,
    KnownModels.Mpt: Mpt,
    KnownModels.GptJ: GptJ,
    KnownModels.Gpt2: Gpt2,
    KnownModels.Bloom: Bloom,
    KnownModels.Llama: Llama
}

@dataclass()
class ModelMetadata():
    """
    A dataclass to store metadata about a model.
    """
    model: KnownModels
    quantization: QuantizationType
    container: ContainerType
    converter:str="llm-rs"
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
    def load_metadata(model_file:Union[str,os.PathLike])->ModelMetadata:
        path = pathlib.Path(model_file)
        if not path.is_file():
            raise ValueError(f"Model file '{model_file}' is not a file!")
        if not path.exists():
            raise ValueError(f"Model file '{model_file}' does not exist!")

        metadata_file = path.with_suffix(".meta")
        if not metadata_file.exists():
            raise ValueError(f"Model file '{model_file}' does not have a metadata file '{metadata_file}'!")
        
        metadata = ModelMetadata.deserialize(json.loads(metadata_file.read_text()))
        return metadata
    
    @staticmethod
    def infer_model_type(model_file:Union[str,os.PathLike])->Type[Model]:
        
        metadata = AutoModel.load_metadata(model_file)
        if metadata.model in _KNOWN_MODELS_MAP:
            return _KNOWN_MODELS_MAP[metadata.model]
        else:
            raise ValueError(f"Unknown model type '{metadata.model}'")
            
        
    @staticmethod
    def load(path:Union[str,os.PathLike],
                  session_config:SessionConfig=SessionConfig(),
                  lora_paths:Optional[List[Union[str,os.PathLike]]]=None,
                  verbose:bool=False)->Model:
        
        model_type = AutoModel.infer_model_type(path)
        return model_type(path,session_config,lora_paths,verbose)
            

class AutoQuantizer():
    """
    Utility to quantize models, without having to specify the model type.
    """
    @staticmethod
    def quantize(model_file:Union[str,os.PathLike],target_path:Optional[Union[str,os.PathLike]]=None,quantization:QuantizationType=QuantizationType.Q4_0,container:ContainerType=ContainerType.GGJT)->Union[str,os.PathLike]:
        metadata=AutoModel.load_metadata(model_file)
        if metadata.quantization != QuantizationType.F16:
            raise ValueError(f"Model '{model_file}' is already quantized to '{metadata.quantization}'")

        model_type = AutoModel.infer_model_type(model_file)

        if target_path is None:
            target_path = os.path.dirname(model_file)
            
        def build_target_name()->str:
            output_path = pathlib.Path(target_path)
            if output_path.is_file():
                return output_path
            else:
                output_path.mkdir(parents=True,exist_ok=True)
                model_path = pathlib.Path(model_file)
                appendix = ""
                if  quantization == QuantizationType.Q4_0:
                    appendix += "-q4_0"
                elif quantization == QuantizationType.Q4_1:
                    appendix += "-q4_1"

                if container == ContainerType.GGJT:
                    appendix += "-ggjt"
            
                filename = model_path.stem.replace("-f16","") + appendix + model_path.suffix
                return str(output_path / filename)

        target_file = build_target_name()
        if pathlib.Path(target_file).exists():
            logging.warning(f"Target file '{target_file}' already exists, skipping quantization")
            return target_file
        
        logging.info(f"Quantizing model '{model_file}' to '{target_file}'")
        model_type.quantize(model_file,target_file,quantization,container)

        metadata_file = pathlib.Path(target_file).with_suffix(".meta")
        quantized_metadata = ModelMetadata(model=metadata.model,quantization=quantization,container=container)
        quantized_metadata.add_hash(target_file)
        logging.info(f"Writing metadata file '{metadata_file}'")
        metadata_file.write_text(json.dumps(quantized_metadata.serialize()))
        logging.info(f"Finished quantizing model '{model_file}' to '{target_file}'")
        return target_file


