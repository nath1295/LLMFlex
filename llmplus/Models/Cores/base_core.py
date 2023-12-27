from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.runnable import RunnableConfig
from typing import Any, List, Dict, Optional, Union, Iterator

class BaseCore:
    """Base class of Core object to store the llm model and tokenizer.
    """
    def __init__(self, model_id: str = 'gpt2', **kwargs) -> None:
        """Initialising the core instance.

        Args:
            model_id (str, optional): Model id (from Huggingface) to use. Defaults to 'gpt2'.
        """
        from transformers import AutoTokenizer
        self._model_id = model_id
        self._core_type = 'BaseCore'
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = None

    @property
    def model(self) -> Any:
        """Model for llms.

        Returns:
            Any: Model for llms.
        """
        return self._model
    
    @property
    def tokenizer(self) -> Any:
        """Tokenizer of the model.

        Returns:
            Any: Tokenizer of the model.
        """
        return self._tokenizer
    
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
        return self._model_id
    
    def encode(self, text: str) -> List[int]:
        """Tokenize the given text.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[int]: List of token ids.
        """
        return self.tokenizer(text=text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Untokenize a list of tokens.

        Args:
            token_ids (List[int]): Token ids to untokenize. 

        Returns:
            str: Untokenized string.
        """
        return self.tokenizer.decode(token_ids=token_ids, skip_special_tokens=True)
    
    def unload(self) -> None:
        """Unload the model from ram."""
        del self._model
        self._model = None
        del self._tokenizer
        self._tokenizer = None
    

class BaseLLM(LLM):
    """Base LLM class for llmplus, using the LLM class from langchain.
    """
    core: BaseCore
    generation_config: Dict[str, Any]
    stop: List[str]

    def __init__(self, core: BaseCore, temperature: float = 0, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, 
                 repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True) -> None:
        """Initialising the LLM.

        Args:
            core (BaseCore): The BaseCore core.
            temperature (float, optional): Set how "creative" the model is, the smaller it is, the more static of the output. Defaults to 0.
            max_new_tokens (int, optional): Maximum number of tokens to generate by the llm. Defaults to 2048.
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

        stop = get_stop_words(stop, tokenizer=core.tokenizer, add_newline_version=stop_newline_version)
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
        output = 'This is a testing llm. '
        tokens = (output * 5).split(' ')

        stream = False
        if 'stream' in kwargs.keys():
            stream = kwargs['stream']

        if stream:
            import time
            def test_stream():
                for i in tokens:
                    time.sleep(0.05)
                    yield i + ' '
            return test_stream()
        
        else:
            return output
        
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
        return 'DebugLLM'
    
    



