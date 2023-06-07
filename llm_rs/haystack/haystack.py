try:
    from haystack.nodes.prompt.invocation_layer.base import PromptModelInvocationLayer
except ImportError:
     raise ImportError(
        'To use the llm_rs.haystack module, please install llm-rs with the additional "haystack" dependencies via: pip install llm-rs[haystack]')

from typing import Dict, List, Union, Type

class RustformersInvocationLayer(PromptModelInvocationLayer):
    def __init__(self, model_name_or_path: str, **kwargs):
        """
        Creates a new PromptModelInvocationLayer instance.

        :param model_name_or_path: The name or path of the underlying model.
        :param kwargs: Additional keyword arguments passed to the underlying model.
        """
        if model_name_or_path is None or len(model_name_or_path) == 0:
            raise ValueError("model_name_or_path cannot be None or empty string")

        self.model_name_or_path = model_name_or_path

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure that length of the prompt and answer is within the maximum token length of the PromptModel.

        :param prompt: Prompt text to be sent to the generative model.
        """
        pass

    def invoke(self, *args, **kwargs):
        """
        It takes a prompt and returns a list of generated text using the underlying model.
        :return: A list of generated text.
        """
        pass

    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Checks if the given model is supported by this invocation layer.

        :param model_name_or_path: The name or path of the model.
        :param kwargs: Additional keyword arguments passed to the underlying model which might be used to determine
        if the model is supported.
        :return: True if this invocation layer supports the model, False otherwise.
        """
        return False