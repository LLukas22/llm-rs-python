from transformers import AutoConfig
import os
from typing import Union
from .models.mpt import MPTAdapter
from .models.gptneox import GptNeoXAdapter
import logging 

class AutoConverter():

    @staticmethod
    def convert(pretrained_model_name_or_path:Union[str,os.PathLike],output_directory:Union[str,os.PathLike])->None:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path,trust_remote_code=True)
        architecture = config.architectures[0]

        adapter=None
        if architecture == "MPTForCausalLM":
            adapter=MPTAdapter
        if architecture == "GPTNeoXForCausalLM":
            adapter=GptNeoXAdapter
        else:
            raise ValueError(f"Unsupported architecture {architecture}")

        logging.info(f"Converting {architecture} via {adapter.__name__} to {output_directory}")
        adapter(pretrained_model_name_or_path).convert(output_directory)