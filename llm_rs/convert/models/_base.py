from abc import ABC, abstractmethod
from typing import Union,Tuple,Any,Optional,BinaryIO
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
import os
import struct
from enum import Enum
import logging
import numpy as np


GGML_MAGIC = 0x67676D6C

class FileTypes(Enum):
    FP32 = 0
    FP16 = 1

def write_magic(out_file:BinaryIO):
    return out_file.write(struct.pack("i", GGML_MAGIC))

def write_file_type(out_file:BinaryIO,file_type:FileTypes=FileTypes.FP16):
    return out_file.write(struct.pack("i", file_type.value))


class BaseAdapter(ABC):
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
        dot_token = self.tokenizer.encode(".")[0]
        for i in range(self.config.vocab_size):
            text = self.tokenizer.decode([dot_token, i]).encode("utf-8")
            # remove the first byte (it's always '.')
            text = text[1:]
            out_file.write(struct.pack("i", len(text)))
            out_file.write(text)


    def _filter_weights(self,name:str,weight:np.ndarray)->bool:
        """Filter weights that should be skipped"""
        return False
    
    def _filter_f16_weights(self,name:str,weight:np.ndarray)->bool:
        """Filter weights that should be stored as fp16"""
        n_dims = len(weight.shape)
        return name.endswith(".weight") and n_dims == 2
    
    def _rename_weights(self,name:str)->str:
        """Rename weights that should be renamed"""
        return name
    
    def _write_weights(self,out_file:BinaryIO):
        weights = self.model.state_dict()
        for name, weight in weights.items():
            data = weight.squeeze().numpy()

            if self._filter_weights(name,data):
                logging.info(f"Skipping {name}")
                continue
            
            name = self._rename_weights(name)
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
            logging.info(f"Writing layer '{name}' with shape {data.shape}")



    def convert(self,output_directory:Union[str,os.PathLike])->str:
        with open(output_directory, "wb") as out_file:
            # write magic
            write_magic(out_file)
            self._write_hyperparameters(out_file)
            write_file_type(out_file)

            self._write_vocabulary(out_file)
            self._write_weights(out_file)
