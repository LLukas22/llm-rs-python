try:
    from langchain.llms.base import LLM
except ImportError:
    raise ImportError(
        'To use the llm_rs.langchain module, please install llm-rs with the additional "langchain" dependencies via: pip install llm-rs[langchain]')

from typing import Any, Dict, Optional, Sequence, Union, List
import os

from pydantic import root_validator
from langchain.callbacks.manager import CallbackManagerForLLMRun

from ..auto import AutoModel, KnownModels
from ..config import GenerationConfig, SessionConfig
from ..base_model import Model

class RustformerLLM(LLM):
    """
    Langchain-Wrapper around a Rustformers model.
    """

    model: Optional[Model] = None #: :meta private:

    model_path_or_repo_id: Union[str,os.PathLike]
    """The path to the model file or directory or the name of a Hugging Face Hub
    model repo."""

    model_type: Optional[KnownModels] = None
    """The model type."""

    model_file: Optional[str] = None
    """The name of the model file in repo or directory."""

    # session_config:SessionConfig=SessionConfig()
    # """Session config for the model."""

    # generation_config:GenerationConfig=GenerationConfig()
    # """Generation config for the model."""

    lora_paths:Optional[List[Union[str,os.PathLike]]]=None
    """Paths to the lora files."""


    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            'model_path_or_repo_id': self.model_path_or_repo_id,
            'model_type': self.model_type,
            'model_file': self.model_file,
            # 'session_config': self.session_config,
            # 'generation_config': self.generation_config,
            'lora_paths': self.lora_paths,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'rustformer'

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate and load model from a local file or remote repo."""
        values['model'] = AutoModel.from_pretrained(
            model_path_or_repo_id= values['model_path_or_repo_id'],
            model_type=values['model_type'],
            model_file=values['model_file'],
            # session_config=values['session_config'],
            lora_paths=values['lora_paths'],
        )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            stop: A list of sequences to stop generation when encountered.

        Returns:
            The generated text.
        """
        text = []
        generation_config = GenerationConfig()

        if stop:
            generation_config.stop_words = list(stop)
        for chunk in self.model.stream(prompt, generation_config=generation_config):
            text.append(chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk, verbose=self.verbose)
        return ''.join(text)