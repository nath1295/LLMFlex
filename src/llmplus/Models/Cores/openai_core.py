import os
from .base_core import BaseCore, BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
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
        api_key = os.environ.get('OPENAI_API_KEY', 'NOAPIKEY') if api_key is None else api_key
        self._model = OpenAI(api_key=api_key, base_url=base_url)
        models = list(map(lambda x: x.id, self._model.models.list().data))
        self._model_id = model_id if model_id is not None else ('gpt-3.5-turbo' if 'gpt-3.5-turbo' in models else models[0])
        self._is_openai = 'gpt-3.5-turbo' in models
        if tokenizer_id is not None:
            from ...utils import get_config
            os.environ['HF_HOME'] = get_config()['hf_home']
            os.environ['TOKENIZERS_PARALLELISM'] = 'true'
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_kwargs)
            self._tokenizer_type = 'transformers'
        elif self._is_openai:
            import tiktoken
            self._tokenizer = tiktoken.encoding_for_model(self._model_id)
            self._tokenizer_type = 'openai'
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
        import warnings
        from .utils import get_stop_words, textgen_iterator
        warnings.filterwarnings('ignore')
        stop = get_stop_words(stop, tokenizer=self.tokenizer, add_newline_version=stop_newline_version, tokenizer_type=self.tokenizer_type)
        if stream:
            def generate():
                for i in self._model.completions.create(
                    model=self.model_id,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=repetition_penalty,
                    max_tokens=max_new_tokens,
                    stop=stop,
                    stream=True
                ):
                    yield i.choices[0].text
            return textgen_iterator(generate(), stop=stop)
        else:
            from langchain.llms.utils import enforce_stop_tokens
            output = self._model.completions.create(
                model=self.model_id,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=repetition_penalty,
                max_tokens=max_new_tokens,
                stop=stop
            ).choices[0].text
            output = enforce_stop_tokens(output, stop=stop)
            return output
