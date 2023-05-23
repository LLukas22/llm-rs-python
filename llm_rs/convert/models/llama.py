from ._base import BaseAdapter,GGJT_MAGIC
from typing import Union,Tuple,Optional,BinaryIO,List,Iterable
from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer,AutoConfig,AutoTokenizer,AutoModelForCausalLM
import os
import torch
import struct
from ...auto import KnownModels
import re
import numpy as np
import logging

#based on https://github.com/ggerganov/llama.cpp/blob/master/convert.py
class LlamaConverter(BaseAdapter):
    model_type:KnownModels=KnownModels.Llama
    file_magic=GGJT_MAGIC

    def load(self,pretrained_model_name_or_path:Union[str,os.PathLike],pretrained_tokenizer_name_or_path:Optional[Union[str,os.PathLike]]=None)->Tuple[AutoConfig,AutoTokenizer,AutoModelForCausalLM]:
        config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
        
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            )
        
        tokenizer_name = pretrained_tokenizer_name_or_path if pretrained_tokenizer_name_or_path else pretrained_model_name_or_path 
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

        def make_tensors_list() -> List[str]:
            ret = [
                'tok_embeddings.weight',
                'norm.weight',
                'output.weight',
            ]
            for i in range(80):  # maximum number of layer
                ret += [
                    f'layers.{i}.attention.wq.weight',
                    f'layers.{i}.attention.wk.weight',
                    f'layers.{i}.attention.wv.weight',
                    f'layers.{i}.attention.wo.weight',
                    f'layers.{i}.attention_norm.weight',
                    f'layers.{i}.feed_forward.w1.weight',
                    f'layers.{i}.feed_forward.w2.weight',
                    f'layers.{i}.feed_forward.w3.weight',
                    f'layers.{i}.ffn_norm.weight',
                ]
            return ret

        self.TENSORS_LIST = make_tensors_list()
        self.TENSORS_SET = set(self.TENSORS_LIST)

        self.mapping = {
            "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.attention.wq.weight",
            "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.attention.wk.weight",
            "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.attention.wv.weight",
            "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.attention.wo.weight",
            "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.feed_forward.w1.weight",
            "model.layers.{i}.mlp.down_proj.weight": "layers.{i}.feed_forward.w2.weight",
            "model.layers.{i}.mlp.up_proj.weight": "layers.{i}.feed_forward.w3.weight",
            "model.layers.{i}.input_layernorm.weight": "layers.{i}.attention_norm.weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.ffn_norm.weight",
        }

        return config,tokenizer,model
    
    def _write_hyperparameters(self,out_file:BinaryIO):
        hyperparameters = self.config.to_dict()
        out_file.write(struct.pack("i", hyperparameters["vocab_size"]))
        out_file.write(struct.pack("i", hyperparameters["hidden_size"]))
        out_file.write(struct.pack("i", 256))
        out_file.write(struct.pack("i", hyperparameters["num_attention_heads"]))
        out_file.write(struct.pack("i", hyperparameters["num_hidden_layers"]))
        out_file.write(struct.pack("i", hyperparameters["hidden_size"]//hyperparameters["num_attention_heads"])) # rot (obsolete)

    def _rename_weights(self, name: str) -> str:
        if name == "model.embed_tokens.weight":
            return "tok_embeddings.weight"
        elif name == "model.norm.weight":
            return "norm.weight"
        elif name == "lm_head.weight":
            return "output.weight"
        
        for old, new in self.mapping.items():
            pattern = old.replace("{i}", "(\d+)")
            match = re.fullmatch(pattern, name)
            if match:
                return new.replace("{i}", match.group(1))
            
        return name
    
    def _transform_weights(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        to_permut = [r"layers.(\d+).attention.wq.weight",r"layers.(\d+).attention.wk.weight"]
        for pattern in to_permut:
            match = re.fullmatch(pattern, name)
            if match:
                n_heads = self.config.num_attention_heads
                numpy_weights:np.ndarray = weight.numpy()
                reshaped = (numpy_weights.reshape(n_heads, 2, numpy_weights.shape[0] // n_heads // 2, *numpy_weights.shape[1:])
                   .swapaxes(1, 2)
                   .reshape(numpy_weights.shape))
                logging.info(f"Permuting {name} from {weight.shape} to {reshaped.shape}")
                return torch.from_numpy(reshaped)
        return weight
    
    def _filter_weights_after_rename(self, name: str, weight: torch.Tensor) -> bool:
            return name not in self.TENSORS_SET
    

    def _write_vocabulary(self,out_file:BinaryIO):
        sentence_piece_tokenizer = self.tokenizer.sp_model

        logging.info(f"Processing vocabulary with size {self.config.vocab_size}")
        def sentencepiece_tokens() -> Iterable[Tuple[bytes, float]]:
            tokenizer = sentence_piece_tokenizer
            for i in range(tokenizer.vocab_size()):
                text: bytes
                if tokenizer.is_unknown(i):
                    text = " \u2047 ".encode("utf-8")
                elif tokenizer.is_control(i):
                    text = b""
                elif tokenizer.is_byte(i):
                    piece = tokenizer.id_to_piece(i)
                    if len(piece) != 6:
                        raise Exception(f"Invalid token: {piece}")
                    byte_value = int(piece[3:-1], 16)
                    text = struct.pack("B", byte_value)
                else:
                    text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
                score: float = tokenizer.get_score(i)
                yield text, score

        for text, score in sentencepiece_tokens():
            out_file.write(struct.pack("i", len(text)))
            out_file.write(text)
            out_file.write(struct.pack("f", score))

    

