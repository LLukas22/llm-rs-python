try:
    from haystack.nodes import PromptModelInvocationLayer
    from haystack.nodes.prompt.invocation_layer import DefaultTokenStreamingHandler
except ImportError:
     raise ImportError(
        'To use the llm_rs.haystack module, please install llm-rs with the additional "haystack" dependencies e.g. via: pip install llm-rs[haystack]')

import os
from typing import Dict, List, Union, Type, Optional
from ..auto import AutoModel,KnownModels
from ..config import QuantizationType,ContainerType,SessionConfig,GenerationConfig
from ..base_model import Model
import logging 

logger = logging.getLogger(__name__)

class RustformersInvocationLayer(PromptModelInvocationLayer):
    def __init__(self, model_name_or_path: Union[str,os.PathLike], 
        max_length: int = 512,
        model_file: Optional[str] = None,
        model_type: Optional[KnownModels] = None,
        session_config:SessionConfig=SessionConfig(),
        tokenizer_path_or_repo_id: Optional[Union[str,os.PathLike]]=None,
        lora_paths:Optional[List[Union[str,os.PathLike]]]=None,
        verbose:bool=False,
        use_hf_tokenizer:bool=True,
        default_quantization:QuantizationType=QuantizationType.Q4_0,
        default_container:ContainerType=ContainerType.GGJT,
        **kwargs):

        """
        Creates a new RustformersInvocationLayer instance.

        :param model_name_or_path: The name or path of the underlying model.
        :param kwargs: See `AutoModel.from_pretrained`.
        """
        if model_name_or_path is None or len(model_name_or_path) == 0:
            raise ValueError("model_name_or_path cannot be None or empty string")

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length 
        self.model:Model = AutoModel.from_pretrained(model_path_or_repo_id = model_name_or_path,
            model_file=model_file,
            model_type=model_type,
            session_config=session_config,
            tokenizer_path_or_repo_id=tokenizer_path_or_repo_id,
            lora_paths=lora_paths,
            verbose=verbose,
            use_hf_tokenizer=use_hf_tokenizer,
            default_quantization=default_quantization,
            default_container=default_container)
        

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure that length of the prompt and answer is within the maximum token length of the PromptModel.

        :param prompt: Prompt text to be sent to the generative model.
        """
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt must be of type str but got {type(prompt)}")
        
        context_length = self.model.config.context_length
        tokenized_prompt = self.model.tokenize(prompt)
        if len(tokenized_prompt) + self.max_length > context_length:
            logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
            "answer length (%s tokens) fit within the max token limit (%s tokens). "
            "Shorten the prompt to prevent it from being cut off",
            len(tokenized_prompt),
            max(0, context_length -  self.max_length),
            self.max_length,
            context_length,
            )
            return self.model.decode(tokenized_prompt[:max(0, context_length -  self.max_length)])

        return prompt

    def invoke(self, *args, **kwargs):
        """
        It takes a prompt and returns a list of generated text using the underlying model.
        :return: A list of generated text.
        """

        prompt = kwargs.pop("prompt",None)

        if not prompt:
            raise ValueError("`prompt` cannot be None")
    
        generation_config = kwargs.pop("generation_config", GenerationConfig())
        generation_config.max_new_tokens = self.max_length

        stream = kwargs.pop("stream", False)
        stream_handler = kwargs.get("stream_handler", DefaultTokenStreamingHandler())

        generated_text = []
        for token in self.model.stream(prompt=prompt,generation_config=generation_config):
            generated_text.append(token)
            if stream:
                stream_handler(token)

        return generated_text

    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Checks if the given model is supported by this invocation layer.

        :param model_name_or_path: The name or path of the model.
        :param kwargs: Additional keyword arguments passed to the underlying model which might be used to determine
        if the model is supported.
        :return: True if this invocation layer supports the model, False otherwise.
        """
        #I guess there is not much to validate here ¯\_(ツ)_/¯
        return model_name_or_path is not None and len(model_name_or_path) > 0