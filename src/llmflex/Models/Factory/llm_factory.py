from __future__ import annotations
from ..Cores.base_core import BaseCore, BaseLLM
from ...Prompts.prompt_template import PromptTemplate
from typing import Literal, Optional, Dict, List, Any, Type

def detect_model_type(model_id: str) -> str:
    """This function attempts to get the model format type with the model id.

    Args:
        model_id (str): Model ID form Huggingface.

    Returns:
        str: Model format type.
    """
    if model_id is None:
        return 'openai'
    model_id = model_id.lower()
    if 'gguf' in model_id:
        return 'gguf'
    elif 'awq' in model_id:
        return 'awq'
    elif 'gptq' in model_id:
        return 'gptq'
    elif 'exl2' in model_id:
        return 'exl2'
    else:
        return 'default'

class LlmFactory:

    def __init__(self, 
                model_id: str, 
                model_type: Literal['auto', 'default', 'gptq', 'awq', 'gguf', 'openai', 'exl2', 'debug'] = 'auto',
                model_file: Optional[str] = None,
                model_kwargs: Optional[Dict[str, Any]] = None,
                revision: Optional[str] = None,
                from_local: bool = False,
                context_length: int = 4096,
                base_url: Optional[str] = None,
                api_key: Optional[str] = None,
                tokenizer_id: Optional[str] = None,
                tokenizer_kwargs: Optional[Dict[str, Any]] = None,
                init_empty: bool = False,
                **kwargs) -> None:
        """Initialise the model core to create LLMs.

        Args:
            model_id (str): Model ID (from Huggingface) to use or the model to use if using OpenAI API core.
            model_type (Literal[&#39;auto&#39;, &#39;default&#39;, &#39;gptq&#39;, &#39;awq&#39;, &#39;gguf&#39;, &#39;openai&#39;, &#39;exl2&#39;, &#39;debug&#39;], optional): Type of model format, if 'auto' is given, model_type will be automatically detected. Defaults to 'auto'.
            model_file (Optional[str], optional): Specific model file to use. Only useful for `model_type="gguf"`. Defaults to None.
            model_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for loading the model. Only useful for Default, GPTQ, and AWQ models. Defaults to None.
            revision (Optional[str], optional): Specific revision of the model repository. Only useful for `model_type="exl2"`. Defaults to None.
            from_local (bool, optional): Whether to treat the model_id given as a local path or a Huggingface ID. Only useful for GGUF models. Defaults to False.
            context_length (int, optional): Size of the context window. Only useful for GGUF models. Defaults to 4096.
            base_url (Optional[str], optional): Base URL for the API. Only useful for OpenAI APIs. Defaults to None.
            api_key (Optional[str], optional): API key for OpenAI API. Defaults to None.
            tokenizer_id (Optional[str], optional): Model ID (from Huggingface) to load the tokenizer. Useful for model types "default", "gptq", "awq", and "openai". Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for loading the tokenizer. Useful for model types "default", "gptq", "awq", and "openai".  Defaults to None.
            init_empty (bool, optional): Initialise without a model core. Should not be used for normal initialisation. Defaults to False.
        """
        if not init_empty:
            self._model_id = model_id
            self._model_type = detect_model_type(model_id=model_id) if model_type=='auto' else model_type
            if self.model_type == 'gguf':
                from ..Cores.llamacpp_core import LlamaCppCore
                self._core = LlamaCppCore(self.model_id, model_file=model_file, context_length=context_length, from_local=from_local, **kwargs)
            elif self.model_type in ['default', 'awq', 'gptq']:
                from ..Cores.huggingface_core import HuggingfaceCore
                self._core = HuggingfaceCore(self.model_id, model_type=self.model_type, model_kwargs=model_kwargs, tokenizer_kwargs=tokenizer_kwargs)
            elif self.model_type == 'openai':
                from ..Cores.openai_core import OpenAICore
                self._core = OpenAICore(base_url=base_url, api_key=api_key, model_id=model_id, tokenizer_id=tokenizer_id, tokenizer_kwargs=tokenizer_kwargs)
                self._model_id = self.core.model_id
            elif self.model_type == 'exl2':
                from ..Cores.exllamav2_core import Exl2Core
                self._core = Exl2Core(self.model_id, revision=revision, **kwargs)
            elif self.model_type == 'debug':
                self._core = BaseCore(model_id=self.model_id, **kwargs)
            else:
                raise ValueError(f'Model type "{self.model_type}" not supported.')
        
    @classmethod
    def from_model_object(cls, model: Any, tokenizer: Any, model_type: Literal['default', 'gptq', 'awq', 'gguf', 'openai', 'exl2'], model_id: str = 'Unknown', **kwargs) -> LlmFactory:
        """Initialise the factory object with an already loaded model.

        Args:
            model (Any): The pre-loaded model.
            tokenizer (Any): The pre-loaded tokenizer.
            model_type (Literal[&#39;default&#39;, &#39;gptq&#39;, &#39;awq&#39;, &#39;gguf&#39;, &#39;openai&#39;, &#39;exl2&#39;]): Type of model format.
            model_id (str, optional): Name to be given to the model, recommend to use the repo ID on HuggingFace. Defaults to 'Unknown'.

        Returns:
            LlmFactory: The initialised llm factory.
        """
        if model_type in ['default', 'gptq', 'awq']:
            from ..Cores.huggingface_core import HuggingfaceCore
            core = HuggingfaceCore.from_model_object(model=model, tokenizer=tokenizer, model_id=model_id, model_type=model_type)
        elif model_type == 'gguf':
            from ..Cores.llamacpp_core import LlamaCppCore
            core = LlamaCppCore.from_model_object(model=model, tokenizer=tokenizer, model_id=model_id)
        elif model_type == 'exl2':
            from ..Cores.exllamav2_core import Exl2Core
            core = Exl2Core.from_model_object(model=model, tokenizer=tokenizer, model_id=model_id)
        elif model_type == 'openai':
            from ..Cores.openai_core import OpenAICore
            core = OpenAICore.from_model_object(model=model, tokenizer=tokenizer, model_id=model_id)
        factory = cls(model_id='', nit_empty=True)
        factory._core = core
        factory._model_id = core.model_id
        factory._model_type = model_type
        return factory

    @property
    def model_id(self) -> str:
        """Model id (from Huggingface).

        Returns:
            str: Model id (from Huggingface).
        """
        return self._model_id
    
    @property
    def model_type(self) -> str:
        """Type of model format.

        Returns:
            str: Type of model format.
        """
        return self._model_type 
    
    @property
    def core(self) -> Type[BaseCore]:
        """Core model of the llm factory.

        Returns:
            Type[BaseCore]: Core model of the llm factory.
        """
        return self._core
    
    @property
    def prompt_template(self) -> PromptTemplate:
        """Default prompt template for the model.

        Returns:
            PromptTemplate: Default prompt template for the model.
        """
        return self.core.prompt_template
    
    def __call__(self, temperature: float = 0.8, max_new_tokens: int = 256, top_p: float = 0.95,
                top_k: int = 40, repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, 
                newline=True, **kwargs: Dict[str, Any]) -> BaseLLM:
        """Calling the object will create a langchain format llm with the generation configurations passed from the arguments. 

        Args:
            temperature (float, optional): Set how "creative" the model is, the samller it is, the more static of the output. Defaults to 0.8.
            max_new_tokens (int, optional): Maximum number of tokens to generate by the llm. Defaults to 256.
            top_p (float, optional): While sampling the next token, only consider the tokens above this p value. Defaults to 0.95.
            top_k (int, optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to 40.
            repetition_penalty (float, optional): The value to penalise the model for generating repetitive text. Defaults to 1.1.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. Defaults to None.
            newline (bool, optional): Whether to add a newline character to the beginning of the "stop" list provided. Defaults to True.

        Returns:
            Type[BaseLLM]: An LLM.
        """
        return self.call(temperature=temperature, max_new_tokens=max_new_tokens,
                          top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                          stop=stop, newline=newline, **kwargs)
    
    def call(self, temperature: float = 0.8, max_new_tokens: int = 2048, top_p: float = 0.95,
                top_k: int = 40, repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, 
                newline=True, **kwargs: Dict[str, Any]) -> Type[BaseLLM]:
        """Calling the object will create a langchain format llm with the generation configurations passed from the arguments. 

        Args:
            temperature (float, optional): Set how "creative" the model is, the samller it is, the more static of the output. Defaults to 0.8.
            max_new_tokens (int, optional): Maximum number of tokens to generate by the llm. Defaults to 2048.
            top_p (float, optional): While sampling the next token, only consider the tokens above this p value. Defaults to 0.95.
            top_k (int, optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to 40.
            repetition_penalty (float, optional): The value to penalise the model for generating repetitive text. Defaults to 1.1.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. Defaults to None.
            newline (bool, optional): Whether to add a newline character to the beginning of the "stop" list provided. Defaults to True.

        Returns:
            BaseLLM: An LLM.
        """
        from ..Cores.base_core import GenericLLM
        return GenericLLM(core=self.core, temperature=temperature, max_new_tokens=max_new_tokens, 
                                 top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, stop=stop, stop_newline_version=newline)
