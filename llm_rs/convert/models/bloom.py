import numpy as np
from ._base import BaseAdapter
from typing import Union,Tuple,Optional,BinaryIO 
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig,BloomForCausalLM
import os
import torch
import struct
import numpy as np
from ...auto import KnownModels

#based on https://github.com/NouamaneTazi/bloomz.cpp/blob/main/convert-hf-to-ggml.py

conv_map = {
    'word_embeddings'       : 'tok_embeddings',
    "word_embeddings_layernorm": 'norm',
        'input_layernorm'        : 'attention_norm',
        'self_attention.query_key_value': 'attention.query_key_value',
        'self_attention.dense':          'attention.wo',
        'post_attention_layernorm': 'ffn_norm',
        'mlp.dense_h_to_4h'           : 'feed_forward.w1',
        'mlp.dense_4h_to_h'           : 'feed_forward.w2',
        'ln_f'                        : 'output_norm',
        'lm_head' : 'output',
        }

class BloomConverter(BaseAdapter):
    model_type:KnownModels=KnownModels.Bloom

    def load(self,pretrained_model_name_or_path:Union[str,os.PathLike],pretrained_tokenizer_name_or_path:Optional[Union[str,os.PathLike]]=None)->Tuple[AutoTokenizer,AutoModelForCausalLM]:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        model = BloomForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            )
        tokenizer_name = pretrained_tokenizer_name_or_path if pretrained_tokenizer_name_or_path else pretrained_model_name_or_path 
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return config,tokenizer,model
    
    def _write_hyperparameters(self,out_file:BinaryIO):
        hyperparameters = self.config.to_dict()
        hyperparameters["multiple_of"] = 1
        out_file.write(struct.pack("i", hyperparameters["vocab_size"]))
        out_file.write(struct.pack("i", hyperparameters["hidden_size"]))
        out_file.write(struct.pack("i", hyperparameters["multiple_of"]))
        out_file.write(struct.pack("i", hyperparameters["n_head"]))
        out_file.write(struct.pack("i", hyperparameters["n_layer"]))

    def _rename_weights(self, name: str) -> str:
        #Arrrrrg i hate this, but upstream is inconsistent with naming
        nn = name
        if name != "lm_head.weight":
            nn = nn.split(".")[1:]
        else:
            nn = nn.split(".")

        if nn[0] == "h":
            nn[0] = "layers"
            mapped = conv_map[".".join(nn[2:-1])]
            name = ".".join(nn[:2] + [mapped] + nn[-1:])
        else:
            mapped = conv_map[".".join(nn[:-1])]
            name = ".".join([mapped] + nn[-1:])
        return name

    def _transform_weights(self, name: str, weight: torch.Tensor) ->  torch.Tensor:
        if "query_key_value" in name:
            q, k, v = weight.reshape(self.config.n_head, 3, -1).unbind(1)
            return torch.cat([q, k, v], dim=0).reshape_as(weight)
        return weight
    
    def _filter_weights(self, name: str, weight: torch.Tensor) -> bool:
        return name.endswith("attn.masked_bias") or name.endswith(".attn.bias") 