from __future__ import annotations
import os
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.callbacks.manager import CallbackManagerForLLMRun
from .base_core import BaseCore, BaseLLM
from typing import Optional, List, Dict, Any, Union, Iterator, Literal

class KeywordsStoppingCriteria(StoppingCriteria):
    '''class for handling stop words in transformers.pipeline'''
    def __init__(self, stop_words: List[str], tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.stopwords = stop_words
        self.stop_ids = list(map(lambda x: self.get_min_ids(x), stop_words))

    def __call__(self, input_ids: Any, scores: Any, **kwargs) -> bool:
        input_list = input_ids[0].tolist()
        for i in self.stop_ids:
            last = len(i)
            if len(input_list) >= last:
                comp = input_list[-last:]
                if comp==i:
                    return True
        return False
    
    def get_min_ids(self, word: str) -> List[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        effective = list()
        for i in range(len(ids)):
            temp = ids[i:]
            text = self.tokenizer.decode(temp)
            if text==word:
                effective.append((text, temp))
            else:
                break
        for i in range(len(ids)):
            temp = ids[:-i]
            text = self.tokenizer.decode(temp)
            if text==word:
                effective.append((text, temp))
            else:
                break
        effective.sort(key=lambda x: len(x[1]))
        return effective[0][1]

class HuggingfaceCore(BaseCore):
    """This is the core class of loading model in awq, gptq, or original format.
    """
    def __init__(self, model_id: str, model_type: Literal['default', 'awq', 'gptq'], 
                 model_kwargs: Optional[Dict[str, Any]] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Initiating the core with transformers.

        Args:
            model_id (str): Model id (from Huggingface) to use.
            model_type (Literal[&#39;default&#39;, &#39;awq&#39;, &#39;gptq&#39;]): Type of model format.
            model_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for loading the model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for loading the tokenizer. Defaults to None.
        """
        self._core_type = 'HuggingfaceCore'
        self._init_config = dict(
            model_id=model_id,
            model_type=model_type,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs
        )

    @classmethod
    def from_model_object(cls, model: Any, tokenizer: Any, model_id: str = 'Unknown', model_type: Literal['default', 'awq', 'gptq'] = 'default') -> HuggingfaceCore:
        """Load a core directly from an already loaded model object and a tokenizer object for the supported formats.

        Args:
            model (Any): The model object.
            tokenizer (Any): The tokenizer object.
            model_id (str): The model_id.
            model_type (Literal['default', 'awq', 'gptq']): The quantize type of the model.

        Returns:
            BaseCore: The initialised core.
        """
        from .utils import get_prompt_template_by_jinja
        core = cls(model_id=model_id, model_type=model_type)
        core._model = model
        core._tokenizer = tokenizer
        core._tokenizer_type = 'transformers'
        core._model_id = model_id
        core._model_type = model_type
        core._prompt_template = get_prompt_template_by_jinja(model_id, tokenizer)
        return core

    def _init_core(self, model_id: str, model_type: Literal['default', 'awq', 'gptq'], 
                   model_kwargs: Optional[Dict[str, Any]] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Initialise everything needed in the core.

        Args:
            model_id (str): The repo ID.
        """
        from ...utils import get_config
        os.environ['HF_HOME'] = get_config()['hf_home']
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._model_id = model_id
        self._model_type = model_type
        model_kwargs = dict() if model_kwargs is None else model_kwargs
        tokenizer_kwargs = dict() if tokenizer_kwargs is None else tokenizer_kwargs
        if not hasattr(tokenizer_kwargs, 'pretrained_model_name_or_path'):
            tokenizer_kwargs['pretrained_model_name_or_path'] = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
        self._tokenizer_type = 'transformers'

        if not hasattr(model_kwargs, 'device_map'):
            model_kwargs['device_map'] = 'auto'
        model_kwargs['pretrained_model_name_or_path'] = model_id
        self._model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    @property
    def model_type(self) -> str:
        """Format of the model.

        Returns:
            str: Format of the model.
        """
        if not hasattr(self, '_model_tpye'):
            self._init_core(**self._init_config)
        return self._model_type
    
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
        from .utils import get_stop_words, textgen_iterator
        import warnings
        warnings.filterwarnings('ignore')
        stop = get_stop_words(stop, tokenizer=self.tokenizer, add_newline_version=stop_newline_version, tokenizer_type=self.tokenizer_type)
        gen_config = dict(
            temperature=temperature if temperature!=0 else 0.01,
            do_sample=temperature!=0,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stopping_criteria=StoppingCriteriaList([KeywordsStoppingCriteria(stop, self.tokenizer)])
        )
        gen_config.update(kwargs)
                
        if stream:
            from threading import Thread
            from transformers import TextIteratorStreamer
            gen_config['streamer'] = TextIteratorStreamer(tokenizer=self.tokenizer, skip_prompt=True)
            
            def pipe(prompt):
                tokens = self.tokenizer(
                    prompt,
                    return_tensors='pt'
                ).input_ids.to(self.model.device)
                output = self.model.generate(tokens, **gen_config)
            
            trd = Thread(target=pipe, args=[prompt])
            def generate():
                trd.start()
                for i in gen_config['streamer']:
                    yield i
                trd.join()
                yield ''
            return textgen_iterator(generate(), stop=stop)
        
        else:
            from langchain.llms.utils import enforce_stop_tokens
            def pipe(prompt):
                tokens = self.tokenizer(
                    prompt,
                    return_tensors='pt'
                ).input_ids.to(self.model.device)
                output = self.model.generate(tokens, **gen_config)
                return self.tokenizer.decode(output[0], skip_special_tokens=True).removeprefix(prompt)

            output = pipe(prompt)
            output = enforce_stop_tokens(output, stop)
            del pipe
            return output
    
    def unload(self) -> None:
        """Unload the model from ram."""
        if not hasattr(self, '_model'):
            return
        device = self._model.device
        del self._model
        self._model = None
        del self._tokenizer
        self._tokenizer = None
        if 'cuda' in device:
            import torch
            torch.cuda.empty_cache()
    