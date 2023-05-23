import numpy as np
from ._base import BaseAdapter
from typing import Union,Tuple,Optional,BinaryIO 
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig,GPTJForCausalLM
import os
import torch
import struct
import numpy as np
from ...auto import KnownModels

#based on https://github.com/ggerganov/ggml/blob/master/examples/gpt-j/convert-h5-to-ggml.py
class GptJConverter(BaseAdapter):
    model_type:KnownModels=KnownModels.GptJ

    def load(self,pretrained_model_name_or_path:Union[str,os.PathLike],pretrained_tokenizer_name_or_path:Optional[Union[str,os.PathLike]]=None)->Tuple[AutoTokenizer,AutoModelForCausalLM]:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        model = GPTJForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            )
        tokenizer_name = pretrained_tokenizer_name_or_path if pretrained_tokenizer_name_or_path else pretrained_model_name_or_path 
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return config,tokenizer,model
    
    def _write_hyperparameters(self,out_file:BinaryIO):
        hyperparameters = self.config.to_dict()
        out_file.write(struct.pack("i", hyperparameters["vocab_size"]))
        out_file.write(struct.pack("i", hyperparameters["n_positions"]))
        out_file.write(struct.pack("i", hyperparameters["n_embd"]))
        out_file.write(struct.pack("i", hyperparameters["n_head"]))
        out_file.write(struct.pack("i", hyperparameters["n_layer"]))
        out_file.write(struct.pack("i", hyperparameters["rotary_dim"]))

    def _write_vocabulary(self, out_file: BinaryIO):
        # write the vocabulary size
        out_file.write(struct.pack("i", self.config.vocab_size))
        return super()._write_vocabulary(out_file)
    

    def _filter_weights(self, name: str, weight: torch.Tensor) -> bool:
        return name.endswith("attn.masked_bias") or name.endswith(".attn.bias") 