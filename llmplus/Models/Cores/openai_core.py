import os
from .base_core import BaseCore
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.runnable import RunnableConfig
from .utils import get_stop_words
from typing import List, Any, Optional, Dict, Union, Iterator

class OpenAICore(BaseCore):
    """Core class for llm models using openai api interface.
    """
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model_id: Optional[str] = None,
                 tokenizer_id: Optional[str] = None, tokenizer_kwargs: Dict[str, Any] = dict()) -> None:
        """Initialising the llm core.

        Args:
            base_url (Optional[str], optional): URL for the model api endpoint, if None is given, it will use the default URL for OpenAI api. Defaults to None.
            api_key (Optional[str], optional): If using OpenAI api, API key should be provided. Defaults to None.
            model_id (Optional[str], optional): If using OpenAI api or using an api with multiple models, please provide the model to use. Otherwise 'gpt-3.5-turbo' or the first available model will be used by default. Defaults to None.
            tokenizer_id (Optional[str], optional): If not using OpenAI api, repo_id to get the tokenizer from HuggingFace must be provided. Defaults to None.
            tokenizer_kwargs (Dict[str, Any], optional): If not using OpenAI api, kwargs can be passed to load the tokenizer from HuggingFace. Defaults to dict().
        """
        from openai import OpenAI
        self._core_type = 'OpenAICore'
        api_key = 'NOAPIKEY' if api_key is None else api_key
        self._model = OpenAI(api_key=api_key, base_url=base_url)
        models = list(map(lambda x: x.id, self._model.models.list().data))
        self._model_id = model_id if model_id is not None else ('gpt-3.5-turbo' if 'gpt-3.5-turbo' in models else models[0])
        self._is_openai = 'gpt-3.5-turbo' in models
        if tokenizer_id is not None:
            from ...utils import get_config
            os.environ['HF_HOME'] = get_config()['hf_home']
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_kwargs)
        elif self._is_openai:
            import tiktoken
            self._tokenizer = tiktoken.encoding_for_model(self._model_id)
        else:
            raise ValueError(f'Cannot infer tokenizer, please specify the tokenizer_id.')
    
    def encode(self, text: str) -> List[int]:
        """Tokenize the given text.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[int]: List of token ids.
        """
        if self._is_openai:
            return self._tokenizer.encode(text)
        return self.tokenizer(text=text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Untokenize a list of tokens.

        Args:
            token_ids (List[int]): Token ids to untokenize. 

        Returns:
            str: Untokenized string.
        """
        if self._is_openai:
            return self._tokenizer.decode(token_ids)
        return self.tokenizer.decode(token_ids=token_ids, skip_special_tokens=True)
    
class OpenAILLM(LLM):
    '''Custom implementation of streaming for models from OpenAI api. Used in the Llm factory to get new llm from the model.'''
    core: OpenAICore
    generation_config: Dict[str, Any]
    stop: List[str]

    def __init__(self, core: OpenAICore, temperature: float = 0, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, 
                 repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True) -> None:
        """Initialising the llm.

        Args:
            core (OpenAICor): The OpenAICore core.
            temperature (float, optional): Set how "creative" the model is, the smaller it is, the more static of the output. Defaults to 0.
            max_new_tokens (int, optional): Maximum number of tokens to generate by the llm. Defaults to 2048.
            top_p (float, optional): While sampling the next token, only consider the tokens above this p value. Defaults to 0.95.
            top_k (int, optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to 40.
            repetition_penalty (float, optional): The value to penalise the model for generating repetitive text. Defaults to 1.1.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. Defaults to None.
            stop_newline_version (bool, optional): Whether to add duplicates of the list of stop words starting with a new line character. Defaults to True.
        """
        tokenizer_type = 'openai' if core._is_openai else 'transformers'
        stop = get_stop_words(stop, core.tokenizer, stop_newline_version, tokenizer_type)

        generation_config = dict(
            temperature = temperature,
            max_new_tokens = max_new_tokens,
            top_p = top_p,
            top_k = top_k,
            repetition_penalty = repetition_penalty
        )

        super().__init__(core=core, generation_config=generation_config, stop=stop)
        self.generation_config = generation_config
        self.core = core
        self.stop = stop

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Dict[str, Any],
    ) -> Union[str, Iterator[str]]:
        """Text generation of the llm. Return the generated string given the prompt. If set `stream=True`, return a python generator that yield the tokens one by one.

        Args:
            prompt (str): The prompt to the llm.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. If provided, it will overide the original llm stop list. Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun], optional): Not used. Defaults to None.

        Returns:
            Union[str, Iterator]: The output string or a python generator, depending on if it's in stream mode.

        Yields:
            Iterator[str]: The next generated token.
        """
        import warnings
        warnings.filterwarnings('ignore')
        tokenizer_type = 'openai' if self.core._is_openai else 'transformers'
        stop = get_stop_words(stop, tokenizer=self.core.tokenizer, add_newline_version=False, tokenizer_type=tokenizer_type) if stop is not None else self.stop
        stream = kwargs.get('stream', False)
        gen_config = self.generation_config.copy()
        gen_config['stop'] = stop
        if stream:
            def generate():
                for i in self.core._model.completions.create(
                    model=self.core.model_id,
                    prompt=prompt,
                    temperature=gen_config['temperature'],
                    top_p=gen_config['top_p'],
                    frequency_penalty=gen_config['repetition_penalty'],
                    max_tokens=gen_config['max_new_tokens'],
                    stop=stop,
                    stream=True
                ):
                    yield i.choices[0].text
            return generate()
        else:
            return self.core._model.completions.create(
                model=self.core.model_id,
                prompt=prompt,
                temperature=gen_config['temperature'],
                top_p=gen_config['top_p'],
                frequency_penalty=gen_config['repetition_penalty'],
                max_tokens=gen_config['max_new_tokens'],
                stop=stop
            ).choices[0].text

    def stream(self, input: str, config: Optional[RunnableConfig] = None, *, stop: Optional[List[str]] = None, **kwargs) -> Iterator[str]:
        """Text streaming of llm generation. Return a python generator of output tokens of the llm given the prompt.

        Args:
            input (str): The prompt to the llm.
            config (Optional[RunnableConfig]): Not used. Defaults to None.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. If provided, it will overide the original llm stop list. Defaults to None.

        Yields:
            Iterator[str]: The next generated token.
        """
        return self._call(prompt=input, stop=stop, stream=True)
    
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
    

    def _llm_type(self) -> str:
        """LLM type.

        Returns:
            str: LLM type.
        """
        return 'OpenAILLM'