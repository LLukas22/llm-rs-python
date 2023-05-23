from ._base import BaseAdapter
from typing import Union,Tuple,Optional,BinaryIO 
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
import os
import torch
import struct
from ...auto import KnownModels

#based on https://github.com/ggerganov/ggml/blob/master/examples/mpt/convert-h5-to-ggml.py
class MPTConverter(BaseAdapter):
    model_type:KnownModels=KnownModels.Mpt
    
    def load(self,pretrained_model_name_or_path:Union[str,os.PathLike],pretrained_tokenizer_name_or_path:Optional[Union[str,os.PathLike]]=None)->Tuple[AutoTokenizer,AutoModelForCausalLM]:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path,trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            )
        
        #MPT uses the `EleutherAI/gpt-neox-20b` tokenizer => use it if no tokenizer is specified
        tokenizer_name = pretrained_tokenizer_name_or_path if pretrained_tokenizer_name_or_path else "EleutherAI/gpt-neox-20b" 
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return config,tokenizer,model
    
    def _write_hyperparameters(self,out_file:BinaryIO):
        hyperparameters = self.config.to_dict()
        out_file.write(struct.pack("i", hyperparameters["d_model"]))
        out_file.write(struct.pack("i", hyperparameters["max_seq_len"]))
        out_file.write(struct.pack("i", hyperparameters["n_heads"]))
        out_file.write(struct.pack("i", hyperparameters["n_layers"]))
        out_file.write(struct.pack("i", hyperparameters["vocab_size"]))
        out_file.write(struct.pack("f", hyperparameters["attn_config"]["alibi_bias_max"]))
        out_file.write(struct.pack("f", hyperparameters["attn_config"]["clip_qkv"] or 0.0))