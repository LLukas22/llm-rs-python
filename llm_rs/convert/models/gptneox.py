import numpy as np
from ._base import BaseAdapter
from typing import Union,Tuple,Optional,BinaryIO 
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
import os
import torch
import struct
import numpy as np
from ...auto import KnownModels

#based on https://github.com/ggerganov/ggml/tree/master/examples/gpt-neox
class GptNeoXConverter(BaseAdapter):
    model_type:KnownModels=KnownModels.GptNeoX

    def load(self,pretrained_model_name_or_path:Union[str,os.PathLike],pretrained_tokenizer_name_or_path:Optional[Union[str,os.PathLike]]=None)->Tuple[AutoTokenizer,AutoModelForCausalLM]:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        model = AutoModelForCausalLM.from_pretrained(
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
        out_file.write(struct.pack("i", hyperparameters["max_position_embeddings"]))
        out_file.write(struct.pack("i", hyperparameters["hidden_size"]))
        out_file.write(struct.pack("i", hyperparameters["num_attention_heads"]))
        out_file.write(struct.pack("i", hyperparameters["num_hidden_layers"]))
        out_file.write(struct.pack("i", int(hyperparameters["rotary_pct"]*(hyperparameters["hidden_size"]//hyperparameters["num_attention_heads"]))))
        out_file.write(struct.pack("i", hyperparameters["use_parallel_residual"] if "use_parallel_residual" in hyperparameters else True))


    def _filter_weights(self, name: str, weight: torch.Tensor) -> bool:
        return name.endswith(".attention.masked_bias") or     \
       name.endswith(".attention.bias") or \
       name.endswith(".attention.rotary_emb.inv_freq")