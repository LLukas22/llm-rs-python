from abc import ABC, abstractmethod
from typing import Union,Tuple,Any,Optional,BinaryIO
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
import os
import struct
from enum import Enum
import logging
import numpy as np
from ...config import QuantizationType,ContainerType
from ...auto_model import ModelMetadata, KnownModels
import pathlib
import json
import torch

GGML_MAGIC = 0x67676D6C

class FileTypes(Enum):
    FP32 = 0
    FP16 = 1

def write_magic(out_file:BinaryIO):
    return out_file.write(struct.pack("i", GGML_MAGIC))

def write_file_type(out_file:BinaryIO,file_type:FileTypes=FileTypes.FP16):
    return out_file.write(struct.pack("i", file_type.value))


class BaseAdapter(ABC):
    model_type:KnownModels=None

    def  __init__(self,pretrained_model_name_or_path:Union[str,os.PathLike],pretrained_tokenizer_name_or_path:Optional[Union[str,os.PathLike]]=None) -> None:
        self.config,self.tokenizer,self.model= self.load(pretrained_model_name_or_path,pretrained_tokenizer_name_or_path)

    @abstractmethod
    def load(self,pretrained_model_name_or_path:Union[str,os.PathLike],pretrained_tokenizer_name_or_path:Optional[Union[str,os.PathLike]]=None)->Tuple[AutoConfig,AutoTokenizer,AutoModelForCausalLM]:
        ...

    @abstractmethod
    def _write_hyperparameters(self,out_file:BinaryIO):
        ...

    def _write_vocabulary(self,out_file:BinaryIO):
        # TODO: temporary hack to not deal with implementing the tokenizer
        logging.info(f"Processing vocabulary with size {self.config.vocab_size}")
        dot_token = self.tokenizer.encode(".")[0]
        for i in range(self.config.vocab_size):
            text = self.tokenizer.decode([dot_token, i]).encode("utf-8")
            # remove the first byte (it's always '.')
            text = text[1:]
            out_file.write(struct.pack("i", len(text)))
            out_file.write(text)


    def _filter_weights(self,name:str,weight:torch.Tensor)->bool:
        """Filter weights that should be skipped"""
        return False
    
    def _filter_f16_weights(self,name:str,data:np.ndarray)->bool:
        """Filter weights that should be stored as fp16"""
        n_dims = len(data.shape)
        return name.endswith(".weight") and n_dims == 2
    
    def _rename_weights(self,name:str)->str:
        """Rename weights that should be renamed"""
        return name
    
    def _transform_weights(self,name:str,weight:torch.Tensor)->torch.Tensor:
        """Transform weights that should be transformed"""
        return weight
    
    def _write_weights(self,out_file:BinaryIO):
        weights = self.model.state_dict()
        for name, weight in weights.items():
            

            if self._filter_weights(name,weight):
                logging.info(f"Skipping layer '{name}'")
                continue
            
            name = self._rename_weights(name)
            weight = self._transform_weights(name,weight)
            data = weight.squeeze().numpy()
            n_dims = len(data.shape);    
            type = FileTypes.FP32

            if self._filter_f16_weights(name,data):
                data = data.astype(np.float16)
                type = FileTypes.FP16
            else:
                data = data.astype(np.float32)

            encoded_name = name.encode("utf-8")
            out_file.write(struct.pack("iii", n_dims, len(encoded_name), type.value))
            for i in range(n_dims):
                out_file.write(struct.pack("i", data.shape[n_dims - 1 - i]))
            out_file.write(encoded_name)
            # data
            data.tofile(out_file)
            logging.info(f"Converted layer '{name}' with shape {data.shape}")



    def convert(self,output_file:Union[str,os.PathLike])->None:

        with open(output_file, "wb") as out_file:
            # write magic
            write_magic(out_file)
            logging.info(f"Processing hyperparameters ...")
            self._write_hyperparameters(out_file)
            write_file_type(out_file)

            self._write_vocabulary(out_file)
            self._write_weights(out_file)
            logging.info(f"Done converting model to GGML format. Saved to '{output_file}'")

        #Create the *.meta file needed for automatic loading
        metadata_file = pathlib.Path(output_file).with_suffix(".meta")
        metadata = ModelMetadata(
            self.model_type,
            QuantizationType.F16,
            ContainerType.GGML,
            "llm-rs")
        metadata.add_hash(output_file)
        metadata_file.write_text(json.dumps(metadata.serialize(),indent=4))
        
        logging.info(f"Created metadata file at '{output_file}'")
        return output_file

