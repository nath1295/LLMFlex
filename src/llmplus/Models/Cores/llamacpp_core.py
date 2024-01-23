import os
from langchain.callbacks.manager import CallbackManagerForLLMRun
from .base_core import BaseCore, BaseLLM
from typing import Optional, List, Dict, Any, Union, Iterator

_chat_formats_map = {
    'llama-2': 'Llama2',
    'vicuna': 'Vicuna1.1',
    'chatml': 'ChatML',
    'openchat': 'OpenChat',
    'zephyr': 'Zephyr'
}

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
    def __init__(self, model_id_or_path: str, model_file: Optional[str] = None, context_length: int = 4096, **kwargs) -> None:
        """Initialising the core.

        Args:
            model_id (str): Model id (from Huggingface) or model file path to use.
            model_file (Optional[str], optional): Specific GGUF model to use. If None, the lowest quant will be used. Defaults to None.
            context_length (int, optional): Context length of the model. Defaults to 4096.
        """
        from ...utils import is_cuda, os_name
        from ...Prompts.prompt_template import PromptTemplate
        self._model_id = os.path.basename(model_id_or_path).removesuffix('.gguf').removesuffix('.GGUF') if model_id_or_path.lower().endswith('.gguf') else model_id_or_path
        self._core_type = 'LlamaCppCore'
        model_dir = get_model_dir(model_id_or_path, model_file=model_file) if not model_id_or_path.lower().endswith('.gguf') else model_id_or_path
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
        self._prompt_template = PromptTemplate.from_preset(_chat_formats_map.get(self._model.chat_format, 'Default'))

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
    
class LlamaCppLLM(BaseLLM):
    '''Custom implementation of streaming for models loaded with `llama-cpp-python`, Used in the Llm factory to get new llm from the model.'''
    core: LlamaCppCore
    generation_config: Dict[str, Any]
    stop: List[str]

    def __init__(self, core: LlamaCppCore, temperature: float = 0, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, 
                 repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True) -> None:
        """Initialising the llm.

        Args:
            core (LlamaCppCore): The LlamaCppCore core.
            temperature (float, optional): Set how "creative" the model is, the smaller it is, the more static of the output. Defaults to 0.
            max_new_tokens (int, optional): Maximum number of tokens to generate by the llm. Defaults to 2048.
            top_p (float, optional): While sampling the next token, only consider the tokens above this p value. Defaults to 0.95.
            top_k (int, optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to 40.
            repetition_penalty (float, optional): The value to penalise the model for generating repetitive text. Defaults to 1.1.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. Defaults to None.
            stop_newline_version (bool, optional): Whether to add duplicates of the list of stop words starting with a new line character. Defaults to True.
        """
        from .utils import get_stop_words
        stop = get_stop_words(stop, core.tokenizer, stop_newline_version, 'llamacpp')

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
        from .utils import get_stop_words
        import warnings
        warnings.filterwarnings('ignore')
        stop = get_stop_words(stop, tokenizer=self.core.tokenizer, add_newline_version=False, tokenizer_type='llamacpp') if stop is not None else self.stop
        stream = kwargs.get('stream', False)
        gen_config = self.generation_config.copy()
        gen_config['stop'] = stop
        for k, v in kwargs.items():
            if k in ['temperature', 'max_new_tokens', 'top_p', 'top_k', 'repetition_penalty']:
                gen_config[k] = v

        if stream:
            def generate():
                for i in self.core.model(
                    prompt=prompt,
                    temperature=gen_config['temperature'],
                    top_k=gen_config['top_k'],
                    top_p=gen_config['top_p'],
                    repeat_penalty=gen_config['repetition_penalty'],
                    max_tokens=gen_config['max_new_tokens'],
                    stop=stop,
                    stream=True
                ):
                    yield i['choices'][0]['text']
            return generate()
        else:
            return self.core.model(
                prompt=prompt,
                temperature=gen_config['temperature'],
                top_k=gen_config['top_k'],
                top_p=gen_config['top_p'],
                repeat_penalty=gen_config['repetition_penalty'],
                max_tokens=gen_config['max_new_tokens'],
                stop=stop
            )['choices'][0]['text']
    
    def _llm_type(self) -> str:
        """LLM type.

        Returns:
            str: LLM type.
        """
        return 'LlamaCppLLM'
