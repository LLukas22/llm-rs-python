from llm_rs import Llama, Mpt, SessionConfig, QuantizationType, ContainerType
from typing import Optional
import sys
from pathlib import Path

base_models=[
    r"E:\GGML_Models\MPT\mpt-7b-f16.bin",
    r"E:\GGML_Models\MPT\mpt-7b-chat-f16.bin",
    r"E:\GGML_Models\MPT\mpt-7b-instruct-f16.bin",
    r"E:\GGML_Models\MPT\mpt-7b-storywriter-f16.bin"
]

formats=[QuantizationType.Q4_0]
containers = [ContainerType.GGJT,ContainerType.GGML]

def build_destination_path(source:str,format:QuantizationType,container:ContainerType)->str:
    path = Path(source)

    container_name = ""
    match container:
        case ContainerType.GGJT:
            container_name = "-ggjt"

    format_name = ""
    match format:
        case QuantizationType.Q4_0:
            format_name = "q4_0"
        case QuantizationType.Q4_1:
            format_name = "q4_1"
    
    new_name = path.stem.replace("-f16","") + f"-{format_name}{container_name}" + path.suffix
    return str(path.parent.joinpath(new_name))


for model_path in base_models:
    for format in formats:
        for container in containers:
            destination_path = build_destination_path(model_path,format,container)
            Mpt.quantize(model_path,destination_path,quantization=format,container=container)




