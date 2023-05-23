from transformers import AutoConfig
import os
from typing import Union
from .models import MPTConverter,GptNeoXConverter,GptJConverter,Gpt2Converter,BloomConverter,LlamaConverter
import logging 
import pathlib


_ARCHITECTURE_CONVERTER_MAP = {
    "MPTForCausalLM": MPTConverter,
    "GPTNeoXForCausalLM": GptNeoXConverter,
    "GPTJForCausalLM": GptJConverter,
    "GPT2LMHeadModel": Gpt2Converter,
    "BloomForCausalLM":BloomConverter,
    "LLaMAForCausalLM":LlamaConverter
}

class AutoConverter():

    @staticmethod
    def convert(pretrained_model_name_or_path:Union[str,os.PathLike],output_path:Union[str,os.PathLike])->str:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path,trust_remote_code=True)
        architecture = config.architectures[0]
        model_name = get_name_from_config(config)

        adapter=None
        if architecture not in _ARCHITECTURE_CONVERTER_MAP:
            raise ValueError(f"Unsupported architecture '{architecture}' for '{model_name}'")
        adapter = _ARCHITECTURE_CONVERTER_MAP[architecture]

        output_file = build_path(output_path,model_name)
        if os.path.exists(output_file):
            logging.warning(f"Skipping {model_name} via {adapter.__name__} because '{output_file}' already exists!")
            return output_file
        
        logging.info(f"Converting {model_name} via {adapter.__name__} to '{output_file}'")
        adapter(pretrained_model_name_or_path).convert(output_file)
        return output_file


def get_name_from_config(config:AutoConfig)->str:
    raw_name:str = ""
    if hasattr(config,"name_or_path"):
        raw_name = config.name_or_path
    elif hasattr(config,"model_type"):
        raw_name = config.model_type
    else:
        raw_name = config.architectures[0]
    return raw_name.split("/")[-1]


def build_path(output_path:Union[str,os.PathLike],model_name:str)->str:
    output_path = pathlib.Path(output_path)
    #User specified a file, so just use that
    if output_path.is_file():
        return output_path

    #User specified a directory => auto generate a file name
    output_path.mkdir(parents=True,exist_ok=True)
    return os.path.join(output_path,f"{model_name}-f16.bin")
    
    