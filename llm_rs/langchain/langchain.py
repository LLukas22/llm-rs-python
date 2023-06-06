try:
    from langchain.llms.base import LLM
    from langchain.embeddings.base import Embeddings
    from langchain.callbacks.manager import CallbackManagerForLLMRun
except ImportError:
    raise ImportError(
        'To use the llm_rs.langchain module, please install llm-rs with the additional "langchain" dependencies via: pip install llm-rs[langchain]')

from typing import Any, Dict, Optional, Sequence, Union, List
import os

from pydantic import root_validator

from ..auto import AutoModel, KnownModels
from ..config import GenerationConfig, SessionConfig
from ..base_model import Model

class RustformersLLM(LLM,Embeddings):
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

    session_config:SessionConfig=SessionConfig()
    """Session config for the model."""

    generation_config:GenerationConfig=GenerationConfig()
    """Generation config for the model."""

    tokenizer_path_or_repo_id: Optional[Union[str,os.PathLike]]=None
    """The path to the tokenizer file or directory or the name of a Hugging Face Hub repo containing the tokenizer."""

    use_hf_tokenizer:bool=True
    """Whether to use the Hugging Face tokenizer or the integrated GGML tokenizer."""

    lora_paths:Optional[List[Union[str,os.PathLike]]]=None
    """Paths to the lora files."""

    verbose:bool=False
    """Whether to print the loading process."""


    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            'model_path_or_repo_id': self.model_path_or_repo_id,
            'model_type': self.model_type,
            'model_file': self.model_file,
            'session_config': self.session_config,
            'generation_config': self.generation_config,
            'tokenizer_path_or_repo_id': self.tokenizer_path_or_repo_id,
            'lora_paths': self.lora_paths,
            'use_hf_tokenizer': self.use_hf_tokenizer,
            'verbose': self.verbose
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'rustformers'

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate and load model from a local file or remote repo."""
        values['model'] = AutoModel.from_pretrained(
            model_path_or_repo_id= values['model_path_or_repo_id'],
            model_type=values['model_type'],
            model_file=values['model_file'],
            session_config=values['session_config'],
            tokenizer_path_or_repo_id= values['tokenizer_path_or_repo_id'],
            lora_paths=values['lora_paths'],
            use_hf_tokenizer=values['use_hf_tokenizer'],
            verbose=values['verbose']
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

        generation_config = self.generation_config
        if stop:
            generation_config.stop_words = list(stop)
        for chunk in self.model.stream(prompt, generation_config=generation_config):
            text.append(chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk, verbose=self.verbose)
        return ''.join(text)
    

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.model.embed(text)