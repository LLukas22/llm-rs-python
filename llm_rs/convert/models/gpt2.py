import numpy as np
from ._base import BaseAdapter
from typing import Union,Tuple,Optional,BinaryIO 
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig,GPT2LMHeadModel
import os
import torch
import struct
import numpy as np
from ...auto import KnownModels
import re

#based on https://github.com/ggerganov/ggml/blob/master/examples/gpt-2/convert-h5-to-ggml.py
class Gpt2Converter(BaseAdapter):
    model_type:KnownModels=KnownModels.Gpt2

    def load(self,pretrained_model_name_or_path:Union[str,os.PathLike],pretrained_tokenizer_name_or_path:Optional[Union[str,os.PathLike]]=None)->Tuple[AutoTokenizer,AutoModelForCausalLM]:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        model = GPT2LMHeadModel.from_pretrained(
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


    def _filter_weights(self, name: str, weight: torch.Tensor) -> bool:
        return name.endswith("attn.masked_bias") or name.endswith(".attn.bias") 
    
    def _rename_weights(self, name: str) -> str:
        new_name=name
        # rename headers to keep compatibility
        # cerebras renames, see https://github.com/ggerganov/ggml/blob/master/examples/gpt-2/convert-cerebras-to-ggml.py
        if name == "transformer.ln_f.weight":
            new_name = "model/ln_f/g"
        elif name == "transformer.ln_f.bias":
            new_name = "model/ln_f/b"
        elif name == "transformer.wte.weight":
            new_name = "model/wte"
        elif name == "transformer.wpe.weight":
            new_name = "model/wpe"
        elif name == "lm_head.weight":
            new_name = "model/lm_head"
        elif re.match(r"transformer.h\.\d+\.ln_1\.weight", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/ln_1/g"
        elif re.match(r"transformer.h\.\d+\.ln_1\.bias", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/ln_1/b"
        elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.weight", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/attn/c_attn/w"
        elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.bias", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/attn/c_attn/b"
        elif re.match(r"transformer.h\.\d+\.attn\.c_proj\.weight", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/attn/c_proj/w"
        elif re.match(r"transformer.h.\d+.attn.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/attn/c_proj/b"
        elif re.match(r"transformer.h.\d+.ln_2.weight", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/ln_2/g"
        elif re.match(r"transformer.h.\d+.ln_2.bias", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/ln_2/b"
        elif re.match(r"transformer.h.\d+.mlp.c_fc.weight", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/mlp/c_fc/w"
        elif re.match(r"transformer.h.\d+.mlp.c_fc.bias", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/mlp/c_fc/b"
        elif re.match(r"transformer.h.\d+.mlp.c_proj.weight", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/mlp/c_proj/w"
        elif re.match(r"transformer.h.\d+.mlp.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            new_name = f"model/h{i}/mlp/c_proj/b"

        return new_name
    
    def _write_vocabulary(self, out_file: BinaryIO):
        # write the vocabulary size
        out_file.write(struct.pack("i", self.config.vocab_size))
        return super()._write_vocabulary(out_file)

    def _transform_weights(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        # for efficiency - transpose these matrices:
        if name.endswith("/mlp/c_proj/w") or name.endswith("/mlp/c_fc/w") or name.endswith("/attn/c_proj/w") or name.endswith("/attn/c_attn/w"):
            #see this issue: https://github.com/pytorch/pytorch/issues/50275
            return weight.transpose(0,1)
        return weight
    
    def _filter_f16_weights(self, name: str, data: np.ndarray) -> bool:
        n_dims = len(data.shape)
        return (name == "model/wte" or name == "model/lm_head" or name[-2:] == "/g" or name[-2:] == "/w") and n_dims == 2

    

        