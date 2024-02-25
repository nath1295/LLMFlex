from __future__ import annotations
from .base_core import BaseCore
from typing import Optional, List, Dict, Any, Union, Iterator


def get_exl2_model_dir(repo_id: str, revision: Optional[str] = None) -> str:
    """Download and get the model repository local directory.

    Args:
        repo_id (str): Huggingface model ID.
        revision (Optional[str], optional): Branch of the repository. If None is given, the main branch will be used. Defaults to None.

    Returns:
        str: Model local directory.
    """
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, revision=revision)

class Exl2Core(BaseCore):

    def __init__(self, model_id: str, revision: Optional[str] = None, **kwargs) -> None:
        """Initialise the exl2 model core.

        Args:
            repo_id (str): Huggingface model ID.
            revision (Optional[str], optional): Branch of the repository. If None is given, the main branch will be used. Defaults to None.
        """
        self._core_type = 'Exl2Core'
        self._init_config = dict(
            model_id=model_id,
            revision=revision
        )

    @classmethod
    def from_model_object(cls, model: Any, tokenizer: Any, model_id: str = 'Unknown', **kwargs) -> Exl2Core:
        """Load a core directly from an already loaded model object and a tokenizer object for the supported formats.

        Args:
            model (Any): The model object.
            tokenizer (Any): The tokenizer object.
            config (Any): The config for initialising cache.
            model_id (str, optional): The model_id. Defaults to "Unknown".

        Returns:
            Exl2Core: The initialised core.
        """
        from exllamav2 import(
            ExLlamaV2Cache,
        )
        from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2BaseGenerator
        cache = ExLlamaV2Cache(model, lazy = True)
        model.load_autosplit(cache)
        core = cls(model_id)
        core._tokenizer = tokenizer
        core._model = dict(
            default=ExLlamaV2BaseGenerator(model, cache, core._tokenizer),
            streamer=ExLlamaV2StreamingGenerator(model, cache, core._tokenizer)
            )
        core._tokenizer_type = 'transformers'
        return core

    def _init_core(self, model_id: str, revision: Optional[str] = None, **kwargs) -> None:
        """Initialise the exl2 model core.

        Args:
            repo_id (str): Huggingface model ID.
            revision (Optional[str], optional): Branch of the repository. If None is given, the main branch will be used. Defaults to None.
        """
        from exllamav2 import(
            ExLlamaV2,
            ExLlamaV2Config,
            ExLlamaV2Cache,
            ExLlamaV2Tokenizer,
        )
        from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2BaseGenerator
        self._model_id = model_id
        config = ExLlamaV2Config()
        config.model_dir = get_exl2_model_dir(model_id, revision)
        config.prepare()

        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, lazy = True)
        model.load_autosplit(cache)

        self._tokenizer = ExLlamaV2Tokenizer(config)
        self._tokenizer_type = 'transformers'
        self._model = dict(
            default=ExLlamaV2BaseGenerator(model, cache, self._tokenizer),
            streamer=ExLlamaV2StreamingGenerator(model, cache, self._tokenizer)
            )

    def encode(self, text: str) -> List[int]:
        """Tokenize the given text.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[int]: List of token ids.
        """
        return self.encode(text).tolist()[0]

    def decode(self, token_ids: List[int]) -> str:
        """Untokenize a list of tokens.

        Args:
            token_ids (List[int]): Token ids to untokenize.

        Returns:
            str: Untokenized string.
        """
        return self.tokenizer.decode_(token_ids, decode_special_tokens=False)
    
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
        from langchain.llms.utils import enforce_stop_tokens
        from exllamav2.generator import ExLlamaV2Sampler
        from .utils import get_stop_words, textgen_iterator
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.token_repetition_penalty = repetition_penalty
        settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])
        settings.top_a = kwargs.get('top_a', 0.0)
        settings.__dict__.update(kwargs)
        stop = get_stop_words(stop, tokenizer=self.tokenizer, add_newline_version=stop_newline_version, tokenizer_type=self.tokenizer_type)

        if stream:
            input_ids = self.tokenizer.encode(prompt)
            self.model['streamer'].warmup()
            self.model['streamer'].set_stop_conditions(stop)
            self.model['streamer'].begin_stream(input_ids, settings)

            def stream_generator():
                cont = True
                generated_tokens = 0
                output = ''
                while cont:
                    chunk, eos, _ = self.model['streamer'].stream()
                    generated_tokens += 1
                    output += chunk
                    if (eos | (generated_tokens == max_new_tokens) | any(s in output for s in stop)):
                        cont = False
                    yield chunk
            return textgen_iterator(stream_generator(), stop)

        else:
            self.model['default'].warmup()
            self.model['default'].set_stop_conditions(stop)

            output = self.model['default'].generate_simple(prompt=prompt, gen_settings=self.settings, num_tokens=max_new_tokens, stop_token=stop)
            output = output.lstrip(prompt)
            return enforce_stop_tokens(output)

    def unload(self) -> None:
        """Unload the model from ram."""
        import gc
        import torch
        device = self._model.device
        del self._model
        self._model = None
        del self._tokenizer
        self._tokenizer = None
        del self._cache
        self._cache = None
        if 'cuda' in device:
            import torch
            torch.cuda.empty_cache()
        gc.collect()
