from __future__ import annotations
import os
from langchain.callbacks.manager import CallbackManagerForLLMRun
from .base_core import BaseCore, BaseLLM
from typing import Optional, List, Dict, Any, Union, Iterator

def get_model_dir(model_id: str, model_file: Optional[str] = None) -> str:
    """Download the model file from Huggingface and get the local directory.

    Args:
        model_id (str): Model's HuggingFace ID.
        model_file (Optional[str], optional): Specific model quant file. If None, will choose the smallest quant automatically. Defaults to None.

    Returns:
        str: Local directory of the model file.
    """
    from huggingface_hub import model_info, hf_hub_download
    from ...utils import get_config
    os.environ['HF_HOME'] = get_config()['hf_home']
    repo = model_info(repo_id=model_id)
    files = list(map(lambda x: x.rfilename, repo.siblings))
    model_files = list(filter(lambda x: x.endswith('.gguf'), files))
    if len(model_files) == 0:
        raise FileNotFoundError(f'No GGUF model files found in this repository "{model_id}".')
    if model_file in model_files:
        pass
    elif model_file is None:
        trial = ['q2', 'q3', 'q4']
        stop = False
        for t in trial:
            for f in model_files:
                if t in f.lower():
                    model_file = f
                    stop = True
                    break
            if stop:
                break
        if stop == False:
            model_file = model_files[0]
    else:
        raise FileNotFoundError(f'File "{model_file}" not found in repository "{model_id}".')
    model_dir = hf_hub_download(repo_id=model_id, filename=model_file)
    return model_dir

class LlamaCppCore(BaseCore):
    """This is the core class of loading model in gguf format.
    """
    def __init__(self, model_id_or_path: str, model_file: Optional[str] = None, context_length: int = 4096, from_local: bool = False, **kwargs) -> None:
        """Initialising the core.

        Args:
            model_id (str): Model id (from Huggingface) or model file path to use.
            model_file (Optional[str], optional): Specific GGUF model to use. If None, the lowest quant will be used. Defaults to None.
            context_length (int, optional): Context length of the model. Defaults to 4096.
            from_local (bool, optional): Whether to treat the model_id given as a local path or a Huggingface ID. Defaults to False.
        """
        self._core_type = 'LlamaCppCore'
        self._init_config = dict(
            model_id_or_path=model_id_or_path,
            model_file=model_file,
            context_length=context_length,
            from_local=from_local
        )
        self._init_config.update(kwargs)

    @classmethod
    def from_model_object(cls, model: Any, tokenizer: Optional[Any] = None, model_id: str = 'Unknown') -> LlamaCppCore:
        """Load a core directly from an already loaded model object.

        Args:
            model (Any): The model object.
            model_id (str): The model_id.
            model_type (Literal['default', 'awq', 'gptq']): The quantize type of the model.

        Returns:
            LlamaCppCore: The initialised core.
        """
        core = cls(model_id_or_path=model_id)
        core._model = model
        core._tokenizer = model
        core._tokenizer_type = 'llamacpp'
        core._model_id = model_id
        return core

    def _init_core(self, model_id_or_path: str, model_file: Optional[str] = None, context_length: int = 4096, from_local: bool = False, **kwargs) -> None:
        """Initialising the core.

        Args:
            model_id (str): Model id (from Huggingface) or model file path to use.
            model_file (Optional[str], optional): Specific GGUF model to use. If None, the lowest quant will be used. Defaults to None.
            context_length (int, optional): Context length of the model. Defaults to 4096.
            from_local (bool, optional): Whether to treat the model_id given as a local path or a Huggingface ID. Defaults to False.
        """
        from ...utils import is_cuda, os_name
        from .utils import detect_prompt_template_by_id
        from ...Prompts.prompt_template import PromptTemplate
        self._model_id = os.path.basename(model_id_or_path).removesuffix('.gguf').removesuffix('.GGUF') if from_local else model_id_or_path
        model_dir = get_model_dir(model_id_or_path, model_file=model_file) if not from_local else model_id_or_path
        from llama_cpp import Llama
        load_kwargs = dict(model_path=model_dir, use_mlock=True, n_ctx=context_length)
        use_gpu = kwargs.get('use_gpu', True if ((is_cuda()) | (os_name() in ['MacOS_apple_silicon'])) else False)
        if use_gpu:
            load_kwargs['n_gpu_layers'] = 50
        try:
            del kwargs['use_gpu']
        except:
            pass
        load_kwargs.update(kwargs)
        self._model = Llama(**load_kwargs)
        self._tokenizer = self._model
        self._tokenizer_type = 'llamacpp'

    def encode(self, text: str) -> List[int]:
        """Tokenize the given text.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[int]: List of token ids.
        """
        return self.tokenizer.tokenize(text.encode())
    
    def decode(self, token_ids: List[int]) -> str:
        """Untokenize a list of tokens.

        Args:
            token_ids (List[int]): Token ids to untokenize. 

        Returns:
            str: Untokenized string.
        """
        return self.tokenizer.detokenize(token_ids).decode()
    
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
        from .utils import get_stop_words
        import warnings
        warnings.filterwarnings('ignore')
        stop = get_stop_words(stop, tokenizer=self.tokenizer, add_newline_version=stop_newline_version, tokenizer_type=self.tokenizer_type)
        gen_config = dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
            stop=stop  
        )
        gen_config.update(kwargs)
        if stream:
            gen_config['stream'] = True
            def generate():
                for i in self.model(
                    prompt=prompt,
                    **gen_config
                ):
                    yield i['choices'][0]['text']
            return generate()
        else:
            gen_config['stream'] = False
            return self.model(
                prompt=prompt,
                **gen_config
            )['choices'][0]['text']
    
