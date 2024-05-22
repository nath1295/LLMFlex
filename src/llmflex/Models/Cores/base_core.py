from __future__ import annotations
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.runnable import RunnableConfig
from ...Prompts.prompt_template import PromptTemplate, DEFAULT_SYSTEM_MESSAGE
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union, Iterator, Type, Tuple, Literal

_chat_formats_map = {
    'llama-2': 'Llama2',
    'vicuna': 'Vicuna',
    'chatml': 'ChatML',
    'openchat': 'OpenChat',
    'zephyr': 'Zephyr'
}

class BaseCore(ABC):
    """Base class of Core object to store the llm model and tokenizer.
    """
    def __init__(self, **kwargs) -> None:
        """Initialising the core instance.

        Args:
            model_id (str, optional): Model id (from Huggingface) to use. Defaults to 'gpt2'.
        """
        self._init_config = kwargs
        self._core_type = 'BaseCore'

    @classmethod
    def from_model_object(cls, model: Any, tokenizer: Any, model_id: str, **kwargs) -> BaseCore:
        """Load a core directly from an already loaded model object and a tokenizer object for the supported formats.

        Args:
            model (Any): The model object.
            tokenizer (Any): The tokenizer object.
            model_id (str): The model_id.

        Returns:
            BaseCore: The initialised core.
        """
        pass

    def _init_core(self, model_id: str, **kwargs) -> None:
        """Initialise everything needed in the core.

        Args:
            model_id (str): The repo ID.
        """
        from transformers import AutoTokenizer
        self._model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._tokenizer_type = 'transformers'
        self._model = None


    @property
    def model(self) -> Any:
        """Model for llms.

        Returns:
            Any: Model for llms.
        """
        if not hasattr(self, '_model'):
            self._init_core(**self._init_config)
        return self._model
    
    @property
    def tokenizer(self) -> Any:
        """Tokenizer of the model.

        Returns:
            Any: Tokenizer of the model.
        """
        if not hasattr(self, '_tokenizer'):
            self._init_core(**self._init_config)
        return self._tokenizer
    
    @property
    def tokenizer_type(self) -> Literal['transformers', 'llamacpp', 'openai']:
        """Type of tokenizer.

        Returns:
            Literal['transformers', 'llamacpp', 'openai']: Type of tokenizer.
        """
        if not hasattr(self, '_tokenizer_type'):
            self._init_core(**self._init_config)
        return self._tokenizer_type
    
    @property
    def core_type(self) -> str:
        """Type of core.

        Returns:
            str: Type of core.
        """
        return self._core_type
    
    @property
    def model_id(self) -> str:
        """Model ID.

        Returns:
            str: Model ID.
        """
        if not hasattr(self, '_model_id'):
            self._init_core(**self._init_config)
        return self._model_id
    
    @property
    def prompt_template(self) -> PromptTemplate:
        """Default prompt template for the model.

        Returns:
            PromptTemplate: Default prompt template for the model.
        """
        if not hasattr(self, '_prompt_template'):
            from transformers import PreTrainedTokenizerBase
            from .utils import detect_prompt_template_by_id, get_prompt_template_by_jinja
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                self._prompt_template = get_prompt_template_by_jinja(self.model_id, tokenizer=self.tokenizer)
            elif self.core_type == 'LlamaCppCore':
                preset = detect_prompt_template_by_id(self.model_id)
                preset = _chat_formats_map.get(self._model.chat_format, 'Default') if preset == 'Default' else preset
                self._prompt_template = PromptTemplate.from_preset(preset)
            else:
                self._prompt_template = PromptTemplate.from_preset(detect_prompt_template_by_id(self.model_id))
        return self._prompt_template
    
    def encode(self, text: str) -> List[int]:
        """Tokenize the given text.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[int]: List of token ids.
        """
        return self.tokenizer(text=text)['input_ids']
    
    def decode(self, token_ids: List[int]) -> str:
        """Untokenize a list of tokens.

        Args:
            token_ids (List[int]): Token ids to untokenize. 

        Returns:
            str: Untokenized string.
        """
        return self.tokenizer.decode(token_ids=token_ids, skip_special_tokens=True)
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, 
                 repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True,
                 stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """Generate the output with the given prompt.

        Args:
            prompt (str): The prompt for the text generation.
            temperature (float, optional): Set how "creative" the model is, the smaller it is, the more static of the output. Defaults to 0.
            max_new_tokens (int, optional): Maximum number of tokens to generate by the llm. Defaults to 2048.
            top_p (float, optional): While sampling the next token, only consider the tokens above this p value. Defaults to 0.95.
            top_k (int, optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to 40.
            repetition_penalty (float, optional): The value to penalise the model for generating repetitive text. Defaults to 1.1.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. Defaults to None.
            stop_newline_version (bool, optional): Whether to add duplicates of the list of stop words starting with a new line character. Defaults to True.
            stream (bool, optional): If True, a generator of the token generation will be returned instead. Defaults to False.

        Returns:
            Union[str, Iterator[str]]: Completed generation or a generator of tokens.
        """
        pass
    
    def unload(self) -> None:
        """Unload the model from ram."""
        import gc
        del self._model
        self._model = None
        del self._tokenizer
        self._tokenizer = None
        gc.collect()
    
class BaseLLM(LLM):
    """Base LLM class, using the LLM class from langchain.
    """
    core: Type[BaseCore]
    generation_config: Dict[str, Any]
    stop: List[str]

    def __init__(self, core: Type[BaseCore], generation_config: Dict[str, Any], stop: List[str]) -> None:
        """Initialising the LLM.

        Args:
            core (Type[BaseCore]): The LLM model core.
            generation_config (Dict[str, Any]): Generation configuration.
            stop (List[str]): List of strings to stop the generation of the llm.
        """
        super().__init__(core=core, generation_config=generation_config, stop=stop)
        self.core = core
        self.generation_config = generation_config
        self.stop = stop

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Dict[str, Any]) -> Union[str, Iterator[str]]:
        """Text generation of the llm. Return the generated string given the prompt. If set "stream=True", return a python generator that yield the tokens one by one.

        Args:
            prompt (str): The prompt to the llm.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. If provided, it will overide the original llm stop list. Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun], optional): Not used. Defaults to None.

        Returns:
            Union[str, Iterator]: The output string or a python generator, depending on if it's in stream mode.

        Yields:
            Iterator[str]: The next generated token.
        """
        from .utils import get_stop_words
        from copy import deepcopy
        default_config = deepcopy(self.generation_config)
        gen_config = dict(
            temperature=kwargs.pop('temperature', default_config.pop('temperature', None)),
            max_new_tokens=kwargs.pop('max_new_tokens', default_config.pop('max_new_tokens', None)),
            top_p=kwargs.pop('top_p', default_config.pop('top_p', None)),
            top_k=kwargs.pop('top_k', default_config.pop('top_k', None)),
            repetition_penalty=kwargs.pop('repetition_penalty', default_config.pop('repetition_penalty', None)),
            stop=self.stop if stop is None else get_stop_words(stop, tokenizer=self.core.tokenizer, 
                                                               add_newline_version=kwargs.pop('stop_newline_version', False), tokenizer_type=self.core.tokenizer_type),
            stream=kwargs.pop('stream', False)
        )
        gen_config.update(default_config)
        gen_config.update(kwargs)
        return self.core.generate(prompt=prompt, **gen_config)
        
    def stream(self, input: str, config: Optional[RunnableConfig] = None, *, stop: Optional[List[str]] = None, **kwargs) -> Iterator[str]:
        """Text streaming of llm generation. Return a python generator of output tokens of the llm given the prompt.

        Args:
            input (str): The prompt to the llm.
            config (Optional[RunnableConfig]): Not used. Defaults to None.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. If provided, it will overide the original llm stop list. Defaults to None.

        Yields:
            Iterator[str]: The next generated token.
        """
        from .utils import get_stop_words
        from copy import deepcopy
        default_config = deepcopy(self.generation_config)
        gen_config = dict(
            temperature=kwargs.pop('temperature', default_config.pop('temperature', None)),
            max_new_tokens=kwargs.pop('max_new_tokens', default_config.pop('max_new_tokens', None)),
            top_p=kwargs.pop('top_p', default_config.pop('top_p', None)),
            top_k=kwargs.pop('top_k', default_config.pop('top_k', None)),
            repetition_penalty=kwargs.pop('repetition_penalty', default_config.pop('repetition_penalty', None)),
            stop=self.stop if stop is None else get_stop_words(stop, tokenizer=self.core.tokenizer, 
                                                               add_newline_version=kwargs.pop('stop_newline_version', False), tokenizer_type=self.core.tokenizer_type),
            stream=True
        )
        gen_config.update(default_config)
        gen_config.update(kwargs)
        gen_config['stream'] = True
        return self.core.generate(prompt=input, **gen_config)
    
    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens given the text string.

        Args:
            text (str): Text

        Returns:
            int: Number of tokens
        """
        return len(self.get_token_ids(text))
    
    def get_token_ids(self, text: str) -> List[int]:
        """Get the token ids of the given text.

        Args:
            text (str): Text

        Returns:
            List[int]: List of token ids.
        """
        return self.core.encode(text=text)
    
    def chat(self, prompt: str, prompt_template: Optional[PromptTemplate] = None, stream: bool = False, 
            system: str = DEFAULT_SYSTEM_MESSAGE, history: Optional[Union[List[str], List[Tuple[str, str]]]] = None, **kwargs) -> Union[str, Iterator[str]]:
        """Chat with the llm given the input.

        Args:
            prompt (str): User message.
            prompt_template (Optional[PromptTemplate], optional): Pormpt template to use. If None is given, the default prompt template will be used. Defaults to None.
            stream (bool, optional): Whether to return the response as an iterator or a string. Defaults to False.
            system (str, optional): System message. Defaults to DEFAULT_SYSTEM_MESSAGE.
            history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): List of conversation history. Defaults to None.

        Returns:
            Union[str, Iterator[str]]: Response of the llm.
        """
        prompt_template = self.core.prompt_template if prompt_template is None else prompt_template
        prompt = prompt_template.create_prompt(prompt, system=system, history=history)
        stop = prompt_template.stop
        if stream:
            return self.stream(prompt, stop=stop, **kwargs)
        else:
            return self.invoke(prompt, stop=stop, **kwargs)
    
    def _llm_type(self) -> str:
        """LLM type.

        Returns:
            str: LLM type.
        """
        return 'BaseLLM'

class GenericLLM(BaseLLM):
    """Generic LLM class, using the LLM class from langchain.
    """
    core: BaseCore
    generation_config: Dict[str, Any]
    stop: List[str]

    def __init__(self, core: Type[BaseCore], temperature: float = 0, max_new_tokens: int = 256, top_p: float = 0.95, top_k: int = 40, 
                 repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True, **kwargs) -> None:
        """Initialising the LLM.

        Args:
            core (Type[BaseCore]): The LLM model core.
            temperature (float, optional): Set how "creative" the model is, the smaller it is, the more static of the output. Defaults to 0.
            max_new_tokens (int, optional): Maximum number of tokens to generate by the llm. Defaults to 256.
            top_p (float, optional): While sampling the next token, only consider the tokens above this p value. Defaults to 0.95.
            top_k (int, optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to 40.
            repetition_penalty (float, optional): The value to penalise the model for generating repetitive text. Defaults to 1.1.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. Defaults to None.
            stop_newline_version (bool, optional): Whether to add duplicates of the list of stop words starting with a new line character. Defaults to True.
        """
        from .utils import get_stop_words
        generation_config = dict(
            temperature = temperature,
            max_new_tokens = max_new_tokens,
            top_p  = top_p,
            top_k = top_k,
            repetition_penalty = repetition_penalty
        )
        generation_config.update(kwargs)

        stop = get_stop_words(stop, tokenizer=core.tokenizer, add_newline_version=stop_newline_version, tokenizer_type=core.tokenizer_type)
        super().__init__(core=core, generation_config=generation_config, stop=stop)
        self.core = core
        self.generation_config = generation_config
        self.stop = stop
    
    def _llm_type(self) -> str:
        """LLM type.

        Returns:
            str: LLM type.
        """
        return 'GenericLLM'
    
    



