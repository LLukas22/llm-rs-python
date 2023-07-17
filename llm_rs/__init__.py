try:
    from .llm_rs import *
except ImportError as e:
    print("DLLs were not boundled with this package. Trying to locate them...")
    import os
    import platform

    #Try to locate CUDA_PATH environment variable
    cuda_path = os.environ.get("CUDA_PATH",None)
    if cuda_path:
        print(f"Found CUDA_PATH environment variable: {cuda_path}")
        if platform.system() == "Windows":
            cuda_path = os.path.join(cuda_path,"bin")
        else:
            cuda_path = os.path.join(cuda_path,"lib64")

        print(f"Adding {cuda_path} to DLL search path...")
        os.add_dll_directory(cuda_path)

    try:
        from .llm_rs import *
    except ImportError as inner_e:
        raise  ImportError("Could not locate DLLs. Please check the documentation for more information.")
   


from .config import GenerationConfig, SessionConfig, Precision, ContainerType, QuantizationType
from .models import Llama, GptJ, Gpt2, Bloom, GptNeoX, Mpt
from .auto import AutoModel, ModelMetadata, KnownModels, AutoQuantizer

__doc__ = llm_rs.__doc__
if hasattr(llm_rs, "__all__"):
    __all__ = llm_rs.__all__