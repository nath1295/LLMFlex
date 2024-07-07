from __future__ import annotations
import os, json
from requests.models import Response
from requests import get, post
from .base_core import BaseCore
from typing import List, Any, Optional, Dict, Union, Iterator

def parse_sse(response: Response) -> Iterator[Dict[str, Any]]:
    """Parsing streaming response from llama server.

    Args:
        response (Response): Response object from the llama server.

    Yields:
        Generator[Dict[str, Any]]: Dictionary with text tokens in the content.
    """
    buffer = ""
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        buffer += chunk
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            event = event.strip()
            if event.startswith("data: "):
                yield json.loads(event[6:])

class OpenAICore(BaseCore):
    """Core class for llm models using openai api interface.
    """
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model_id: Optional[str] = None,
                 tokenizer_id: Optional[str] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Initialising the llm core.

        Args:
            base_url (Optional[str], optional): URL for the model api endpoint, if None is given, it will use the default URL for OpenAI api. Defaults to None.
            api_key (Optional[str], optional): If using OpenAI api, API key should be provided. Defaults to None.
            model_id (Optional[str], optional): If using OpenAI api or using an api with multiple models, please provide the model to use. Otherwise 'gpt-3.5-turbo' or the first available model will be used by default. Defaults to None.
            tokenizer_id (Optional[str], optional): If not using OpenAI api, repo_id to get the tokenizer from HuggingFace must be provided. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): If not using OpenAI api, kwargs can be passed to load the tokenizer from HuggingFace. Defaults to None.
        """
        self._core_type = 'OpenAICore'
        self._init_config = dict(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            tokenizer_kwargs=tokenizer_kwargs
        )

    @classmethod
    def from_model_object(cls, model: Any, tokenizer: Any, model_id: Optional[str] = None, **kwargs) -> OpenAICore:
        """Load a core directly from an already loaded model object and a tokenizer object for the supported formats.

        Args:
            model (Any): The model object.
            tokenizer (Any): The tokenizer object.
            model_id (Optional[str], optional): The model_id. Defaults to None.

        Returns:
            OpenAICore: The initialised core.
        """
        from transformers import PreTrainedTokenizerBase
        from .utils import get_prompt_template_by_jinja
        core = cls()
        core._model = model
        core._tokenizer = tokenizer
        models = list(map(lambda x: x.id, core._model.models.list().data))
        core._model_id = model_id if model_id is not None else ('gpt-3.5-turbo' if 'gpt-3.5-turbo' in models else models[0])
        core._is_openai = 'gpt-3.5-turbo' in models
        core._tokenizer_type = 'transformers' if isinstance(tokenizer, PreTrainedTokenizerBase) else 'openai'
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            core._prompt_template = get_prompt_template_by_jinja(core.model_id, tokenizer)
        return core

        
    def _init_core(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model_id: Optional[str] = None,
                 tokenizer_id: Optional[str] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Initialising the llm core.

        Args:
            base_url (Optional[str], optional): URL for the model api endpoint, if None is given, it will use the default URL for OpenAI api. Defaults to None.
            api_key (Optional[str], optional): If using OpenAI api, API key should be provided. Defaults to None.
            model_id (Optional[str], optional): If using OpenAI api or using an api with multiple models, please provide the model to use. Otherwise 'gpt-3.5-turbo' or the first available model will be used by default. Defaults to None.
            tokenizer_id (Optional[str], optional): If not using OpenAI api, repo_id to get the tokenizer from HuggingFace must be provided. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): If not using OpenAI api, kwargs can be passed to load the tokenizer from HuggingFace. Defaults to None.
        """
        from openai import OpenAI
        api_key = os.environ.get('OPENAI_API_KEY', 'NOAPIKEY') if api_key is None else api_key
        self._model = OpenAI(api_key=api_key, base_url=base_url)
        self._openai_url = base_url.removesuffix('/') if base_url is not None else base_url
        self._llama_url = self._openai_url.removesuffix('/v1') if self._openai_url is not None else self._openai_url
        models = list(map(lambda x: x.id, self._model.models.list().data))
        self._model_id = model_id if model_id is not None else ('gpt-3.5-turbo' if 'gpt-3.5-turbo' in models else models[0])
        self._is_openai = 'gpt-3.5-turbo' in models
        if tokenizer_id is not None:
            from ...utils import get_config
            os.environ['HF_HOME'] = get_config()['hf_home']
            os.environ['TOKENIZERS_PARALLELISM'] = 'true'
            from transformers import AutoTokenizer
            tokenizer_kwargs = dict() if tokenizer_kwargs is None else tokenizer_kwargs
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_kwargs)
            self._tokenizer_type = 'transformers'
        elif self._is_openai:
            import tiktoken
            self._tokenizer = tiktoken.encoding_for_model(self._model_id)
            self._tokenizer_type = 'openai'
        else:
            raise ValueError(f'Cannot infer tokenizer, please specify the tokenizer_id.')
        
    @property
    def base_url(self) -> str:
        """The base url of the API.

        Returns:
            str: The base url of the API.
        """
        return str(self.model.base_url._uri_reference)
    
    @property
    def is_llama(self) -> bool:
        """Whether or not the server is a llama.cpp server.

        Returns:
            bool: Whether or not the server is a llama.cpp server.
        """
        if not hasattr(self, '_is_llama'):
            self.base_url
            if self._llama_url is None:
                self._is_llama = False
            else:
                health = get(self._llama_url + '/health').text
                try:
                    health = json.loads(health)
                    health = health.get('status')
                    self._is_llama = health is not None
                except:
                    self._is_llama = False
        return self._is_llama
    
    def encode(self, text: str) -> List[int]:
        """Tokenize the given text.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[int]: List of token ids.
        """
        if self._is_openai:
            return self._tokenizer.encode(text)
        return self.tokenizer(text=text)['input_ids']
    
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
        if self.is_llama:
            return self.llama_generate(prompt=prompt, 
            temperature=temperature, max_new_tokens=max_new_tokens, 
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, stop=stop, stop_newline_version=stop_newline_version, stream=stream, **kwargs)
        import warnings
        from .utils import get_stop_words, textgen_iterator
        warnings.filterwarnings('ignore')
        stop = get_stop_words(stop, tokenizer=self.tokenizer, add_newline_version=stop_newline_version, tokenizer_type=self.tokenizer_type)
        gen_config = dict(
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
            stop=stop
        )
        gen_config.update(kwargs)
        if stream:
            gen_config['stream'] = True
            def generate():
                for i in self.model.completions.create(
                    model=self.model_id,
                    prompt=prompt,
                    **gen_config
                ):
                    yield i.choices[0].text
            return textgen_iterator(generate(), stop=stop)
        else:
            from langchain.llms.utils import enforce_stop_tokens
            gen_config['stream'] = False
            output = self.model.completions.create(
                model=self.model_id,
                prompt=prompt,
                **gen_config
            )
            output = output.choices[0].text
            output = enforce_stop_tokens(output, stop=stop)
            return output

    def llama_generate(self, prompt: str, temperature: float = 0, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, 
                 repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True,
                 stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """Generate the output with the given prompt for llama.cpp server.

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
        import warnings
        from .utils import get_stop_words, textgen_iterator
        warnings.filterwarnings('ignore')
        stop = get_stop_words(stop, tokenizer=self.tokenizer, add_newline_version=stop_newline_version, tokenizer_type=self.tokenizer_type)
        gen_config = dict(
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            n_predict=max_new_tokens,
            stop=stop,
            cache_prompt=True
        )
        gen_config.update(kwargs)
        if stream:
            gen_config['stream'] = True
            def generate():
                headers = {
                    "Content-Type": "application/json"
                }
                gen_config['prompt'] = prompt
                response = post(self._llama_url + '/completion', data=json.dumps(gen_config), headers=headers, stream=True)
                for i in parse_sse(response=response):
                    yield i['content']
            return textgen_iterator(generate(), stop=stop)
        else:
            from .utils import enforce_stop_tokens
            gen_config['stream'] = False
            gen_config['prompt'] = prompt
            headers = {
                "Content-Type": "application/json"
            }
            output = post(self._llama_url + '/completion', data=json.dumps(gen_config), headers=headers).text
            output = json.loads(output)['content']
            output = enforce_stop_tokens(output, stop=stop)
            return output