from .config import QuantizationType,ContainerType,SessionConfig
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
from huggingface_hub import snapshot_download
from huggingface_hub.utils import validate_repo_id, HFValidationError
from huggingface_hub import cached_assets_path

class QuantizationVersions(Enum):
    Not_Quantized=auto()
    V1=auto()
    V2=auto()

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


class PathType(Enum):
    DIR = auto()
    FILE = auto()
    REPO = auto()
    UNKNOWN = auto() 

def _get_path_type(path: Union[str,os.PathLike]) -> PathType:
    p = pathlib.Path(path)
    if p.is_file():
        return PathType.FILE
    elif p.is_dir():
        return PathType.DIR
    try:
        validate_repo_id(path)
        return PathType.REPO
    except HFValidationError:
        pass
    return PathType.UNKNOWN


@dataclass()
class ModelMetadata():
    """
    A dataclass to store metadata about a model.
    """
    model: KnownModels
    quantization: QuantizationType
    container: ContainerType
    quantization_version: QuantizationVersions=QuantizationVersions.Not_Quantized
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
            "quantization_version": self.quantization_version.name,
            "container": repr(self.container).split(".")[-1],
            "converter": self.converter,
            "hash": self.hash
        }
    
    @staticmethod
    def deserialize(metadata_dict):
        return ModelMetadata(
            model = KnownModels[metadata_dict["model"]],
            quantization = _QUANTIZATION_TYPE_MAP[metadata_dict["quantization"]],
            quantization_version= QuantizationVersions[metadata_dict["quantization_version"]],
            container = _CONTAINER_TYPE_MAP[metadata_dict["container"]],
            converter = metadata_dict["converter"],
            hash = metadata_dict["hash"]
        )


@dataclass
class AutoConfig():
    repo_type:Optional[str] = None
    model_type: Optional[str] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_repo_id: Union[str, os.PathLike],
        **kwargs,
    ) -> 'AutoConfig':
        path_type = _get_path_type(model_path_or_repo_id)
        if path_type == PathType.UNKNOWN:
            raise ValueError(
                f"Model path '{model_path_or_repo_id}' doesn't exist.")
        elif path_type == PathType.FILE:
            raise ValueError(
                f"Model path '{model_path_or_repo_id}' is a file. "
                "Please provide a directory or a repo id.")
        
        auto_config = AutoConfig()
        if path_type == PathType.DIR:
            cls._update_from_dir(model_path_or_repo_id, auto_config)
        elif path_type == PathType.REPO:
            cls._update_from_repo(model_path_or_repo_id, auto_config)

        return auto_config

    @classmethod
    def _update_from_repo(
        cls,
        repo_id: str,
        auto_config: 'AutoConfig',
    ) -> None:
        path = snapshot_download(repo_id=repo_id, allow_patterns='config.json')
        cls._update_from_dir(path, auto_config)

    @classmethod
    def _update_from_dir(cls, path: str, auto_config: 'AutoConfig') -> None:
        path = (pathlib.Path(path) / 'config.json').resolve()
        if path.is_file():
            cls._update_from_file(path, auto_config)
        else:
            raise ValueError(f"Config path '{path}' doesn't exist.")

    @classmethod
    def _update_from_file(cls, path: str, auto_config: 'AutoConfig') -> None:
        with open(path) as f:
            config = json.load(f)
        auto_config.model_type = config.get('model_type')
        auto_config.repo_type = config.get('repo_type')

class AutoModel():
    """
    Utility to load models, without having to specify the model type.
    """
    @classmethod
    def load_metadata(cls,model_file:Union[str,os.PathLike])->ModelMetadata:
        path = pathlib.Path(model_file)
        if not path.is_file():
            raise ValueError(f"Model file '{model_file}' is not a file!")
        if not path.exists():
            raise ValueError(f"Model file '{model_file}' does not exist!")

        metadata_file = path.with_suffix(".meta")
        if not metadata_file.exists():
            raise ValueError(f"Model file '{model_file}' does not have a metadata file '{metadata_file}'! If you want to autoload this model, please specify the model type.")
        
        metadata = ModelMetadata.deserialize(json.loads(metadata_file.read_text()))
        return metadata
    
    @classmethod
    def _infer_model_type(cls,model_file:Union[str,os.PathLike],known_model:Optional[KnownModels]=None)->Type[Model]:
        model_to_lookup = None
        if known_model:
            model_to_lookup = known_model
        else:
            metadata = cls.load_metadata(model_file)
            model_to_lookup = metadata.model

        if model_to_lookup in _KNOWN_MODELS_MAP:
            return _KNOWN_MODELS_MAP[model_to_lookup]
        else:
            raise ValueError(f"Unknown model type '{model_to_lookup}'")
            
        
    @classmethod
    def from_file(cls, path:Union[str,os.PathLike],
                  model_type: Optional[KnownModels] = None,
                  session_config:SessionConfig=SessionConfig(),
                  lora_paths:Optional[List[Union[str,os.PathLike]]]=None,
                  verbose:bool=False)->Model:
        
        model = cls._infer_model_type(path,model_type)
        return model(path,session_config,lora_paths,verbose)
    
    @classmethod
    def from_pretrained(cls,
        model_path_or_repo_id: Union[str,os.PathLike],
        model_file: Optional[str] = None,
        model_type: Optional[KnownModels] = None,
        session_config:SessionConfig=SessionConfig(),
        lora_paths:Optional[List[Union[str,os.PathLike]]]=None,
        verbose:bool=False,
        default_quantization:QuantizationType=QuantizationType.Q4_0,
        default_container:ContainerType=ContainerType.GGJT)->Model:

        path_type = _get_path_type(model_path_or_repo_id)

        if path_type == PathType.UNKNOWN:
            raise ValueError(f"Unknown path type for '{model_path_or_repo_id}'")
        elif path_type == PathType.FILE:
            return cls.from_file(model_path_or_repo_id,model_type,session_config,lora_paths,verbose)
        else:
            if path_type == PathType.REPO:

                try: 
                    config = AutoConfig.from_pretrained(
                        model_path_or_repo_id,
                    )
                except ValueError:
                    logging.warning("Could not find config.json in repo, assuming GGML model...")
                    config = AutoConfig(repo_type="GGML")
                       

                if config.repo_type != "GGML":
                    logging.warning("Found normal HuggingFace model, starting conversion...")
                    return cls.from_transformer(model_path_or_repo_id, session_config, lora_paths, verbose, default_quantization, default_container)
            
                resolved_path = cls._find_model_path_from_repo(model_path_or_repo_id,model_file)
                return cls.from_file(resolved_path,model_type,session_config,lora_paths,verbose)
            
            elif path_type == PathType.DIR:
                resolved_path = cls._find_model_path_from_dir(model_path_or_repo_id,model_file)
                return cls.from_file(resolved_path,model_type,session_config,lora_paths,verbose)


    @classmethod
    def _find_model_path_from_dir(
        cls,
        directory: str,
        filename: Optional[str] = None,
    ) -> str:
        path = pathlib.Path(directory).resolve()
        if filename:
            file = (path / filename)
            if not file.is_file():
                raise ValueError(
                    f"Model file '{filename}' not found in '{path}'")
            return str(file)

        files = [
            f for f in path.iterdir()
            if f.is_file() and f.name.endswith('.bin')
        ]

        if len(files) == 0:
            raise ValueError(f"No model files found in '{path}'")
        elif len(files) > 1:
            raise ValueError(
                f"Multiple model files found in '{path}'! Please specify one of the following model files to load: '{','.join([f.name for f in files])}'"
            )

        return str(files[0].resolve())
    
    @classmethod
    def _find_model_path_from_repo(
        cls,
        repo_id: str,
        filename: Optional[str] = None,
    ) -> str:
        cache_directory = cached_assets_path("llm-rs",namespace=repo_id)
        #we dont want to download the whole repo, just the metadata
        directory = snapshot_download(repo_id=repo_id,
                                allow_patterns='*.meta',
                                local_dir=cache_directory)
        path = pathlib.Path(directory).resolve()
        files = [
            f for f in path.iterdir()
            if f.is_file() and f.name.endswith('.meta')
        ]

        if len(files) == 0 and filename is None:
            raise ValueError(f"No model files found in '{repo_id}'! Either specify the model file via the `model_file` parameter or make sure the repository contains a *.meta files for each model.")
        if len(files) > 1 and filename is None:
            raise ValueError(
                f"Multiple model files found in '{repo_id}'! Please specify one of the following model files to load: '{','.join([f.name.replace('.meta','.bin') for f in files])}' "
            )
        #if we have a filename, we can just download that file
        filename = filename if filename else files[0].name.replace('.meta','.bin')
        model_directory = snapshot_download(repo_id=repo_id,
                                allow_patterns=filename,
                                local_dir=cache_directory)
        
        return cls._find_model_path_from_dir(model_directory, filename=filename)
    
    @classmethod
    def from_transformer(cls,
        model_path_or_repo_id: Union[str,os.PathLike],
        session_config:SessionConfig=SessionConfig(),
        lora_paths:Optional[List[Union[str,os.PathLike]]]=None,
        verbose:bool=False,
        default_quantization:QuantizationType=QuantizationType.Q4_0,
        default_container:ContainerType=ContainerType.GGJT):
        
        try:
            from .convert import AutoConverter
            from .convert.auto_converter import get_name_from_config
            from transformers import AutoConfig
        except ImportError:
            raise ImportError("Model conversion needs additional dependencies. Please install 'llm-rs[convert]' to use this functionality!")
        

        config = AutoConfig.from_pretrained(model_path_or_repo_id,trust_remote_code=True)
        model_name = get_name_from_config(config)
        export_path = cached_assets_path("llm-rs",namespace=model_name)
        converted_model = AutoConverter.convert(model_path_or_repo_id,export_path)
        if default_quantization != QuantizationType.F16:
            converted_model = AutoQuantizer.quantize(converted_model,quantization=default_quantization,container=default_container)
        return cls.from_file(converted_model,None,session_config,lora_paths,verbose)
    
class AutoQuantizer():
    """
    Utility to quantize models, without having to specify the model type.
    """
    @staticmethod
    def quantize(model_file:Union[str,os.PathLike],target_path:Optional[Union[str,os.PathLike]]=None,quantization:QuantizationType=QuantizationType.Q4_0,container:ContainerType=ContainerType.GGJT)->Union[str,os.PathLike]:
        metadata=AutoModel.load_metadata(model_file)
        if metadata.quantization != QuantizationType.F16:
            raise ValueError(f"Model '{model_file}' is already quantized to '{metadata.quantization}'")

        model_type = AutoModel._infer_model_type(model_file)

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
        quantized_metadata = ModelMetadata(model=metadata.model,quantization=quantization,container=container,quantization_version=QuantizationVersions.V1)
        quantized_metadata.add_hash(target_file)
        logging.info(f"Writing metadata file '{metadata_file}'")
        metadata_file.write_text(json.dumps(quantized_metadata.serialize()))
        logging.info(f"Finished quantizing model '{model_file}' to '{target_file}'")
        return target_file


