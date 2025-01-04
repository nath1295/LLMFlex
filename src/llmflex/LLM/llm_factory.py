from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from .Engine.base_engine import BaseEngine, BaseLLM
    from ..Tokenizer.base_tokenizer import BaseTokenizer
    from ..Prompt.chat_template import ChatTemplate

class LLMFactory:
    """Class for creating LLMs with different default generation settings from the same LLM engine.
    """
    def __init__(self, engine: "BaseEngine") -> None:
        """Initialises the class with the underlying LLM engine.

        Args:
            engine (BaseEngine): The underlying LLM engine.
        """
        self._engine = engine

    @property
    def engine(self) -> "BaseEngine":
        """Engine used in the LLMs.

        Returns:
            BaseEngine: Engine used in the LLMs.
        """
        return self._engine
    
    @property
    def engine_class(self) -> str:
        """The class name of the LLM engine.

        Returns:
            str: The class name of the LLM engine.
        """
        return self.engine.__class__.__name__
    
    @property
    def tokenizer(self) -> "BaseTokenizer":
        """Returns the tokenizer used in the LLM.

        Returns:
            BaseTokenizer: The tokenizer used in the LLM.
        """
        return self.engine.tokenizer
    
    @property
    def chat_template(self) -> "ChatTemplate":
        """Returns the default chat template for the LLM.

        Returns:
            ChatTemplate: The default chat template for the LLM.
        """
        return self.engine.chat_template
    
    @property
    def model_id(self) -> str:
        """Returns the ID of the LLM model.

        Returns:
            str: The ID of the LLM model.
        """
        return self.engine.model_id
    
    @property
    def model_name(self) -> str:
        """Returns the name of the LLM model.

        Returns:
            str: The name of the LLM model.
        """
        return self.engine.model_name
    
    def __call__(self,
            temperature: float = 0.0,
            max_new_tokens: int = 512,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            top_k: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            stop: Optional[List[str]] = None,
            seed: Optional[int] = None,
            **kwargs
        ) -> "BaseLLM":
        """Get a new LLM with the given generation arguments as default arguments.

        Args:
            temperature (float, optional): The temperature to use for generation. Defaults to 0.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
            top_p (Optional[float], optional): The top-p parameter for generation. Defaults to None.
            min_p (Optional[float], optional): The min-p parameter for generation. Defaults to None.
            top_k (Optional[int], optional): The top-k parameter for generation. Defaults to None.
            repetition_penalty (Optional[float], optional): The repetition penalty for generation. Defaults to None.
            stop (Optional[List[str]], optional): A list of stop words to end generation. Defaults to None.
            seed (Optional[int], optional): The seed for generation. Defaults to None.
            kwargs: Other extra keywowrd arguments.
        """
        from .Engine.base_engine import BaseLLM
        llm = BaseLLM(
            engine=self.engine,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop,
            seed=seed,
            **kwargs
        )
        return llm
    
    @classmethod
    def from_llamacpp(cls,
            pretrained_model_name_or_path: str, 
            model_file: Optional[str] = None,
            tokenizer_name_or_path: Optional[str] = None,
            context_size: int = 8192,
            verbose: bool = False,
            download_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> LLMFactory:
        """Initializes the llm factory with llama-cpp-python.

        Args:
            pretrained_model_name_or_path (str): The HuggingFace repository name or the full path of the model file.
            model_file (str, optional): The model filename if a HuggingFace repository name is given for tokenizer_id_or_path. Defaults to None.
            tokenizer_name_or_path (str, optional): The Huggingface repository name or the full path to load the HuggingFace tokenizer. If not provided, structured text generation might encounter problems. Defaults to None.
            context_size (int, optional): The size of the context window. Defaults to 8192.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            download_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the download function. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the Llama class.
        """
        from .Engine.llamacpp_engine import LlamaCppEngine
        engine = LlamaCppEngine(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_file=model_file,
            tokenizer_name_or_path=tokenizer_name_or_path,
            context_size=context_size,
            verbose=verbose,
            download_kwargs=download_kwargs,
            **kwargs
        )
        return cls(engine)
    
    @classmethod
    def from_mlx(cls,
            pretrained_model_name_or_path: str,
            quantization: Literal['fp16', 'q8', 'q4', 'q2'] = 'fp16',
            tokenizer_name_or_path: Optional[str] = None,
            revision: Optional[str] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            prefill_step_size: int = 512,
            **kwargs
        ) -> LLMFactory:
        """Initializes the llm factory with mlx-textgen.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained model to use.
            quantization (Literal['fp16', 'q8', 'q4', 'q2'], optional): The quantization method to use for the model. Defaults to 'fp16'.
            tokenizer_name_or_path (Optional[str], optional): The name or path of the tokenizer to use. Defaults to None.
            revision (Optional[str], optional): The branch of the Huggingface repository to use. Defaults to None.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
            prefill_step_size (int, optional): The batch size of prompt processing. Defaults to 512.
        """
        from .Engine.mlx_engine import MLXEngine
        engine = MLXEngine(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            quantization=quantization,
            tokenizer_name_or_path=tokenizer_name_or_path,
            revision=revision,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            prefill_step_size=prefill_step_size,
            **kwargs
        )
        return cls(engine)
    
    @classmethod
    def from_openai(cls,
            model_id: Optional[str],
            tokenizer_name_or_path: Optional[str], 
            tokenizer_kwargs: Optional[Dict[str, Any]] = None, 
            base_url: Optional[str] = None,
            api_engine: Optional[Literal['openai', 'mlx-textgen', 'vllm', 'llama.cpp', 'llama-cpp-python']] = None,
            api_key: Optional[str] = None,
            **kwargs
        ) -> LLMFactory:
        """Initializes the llm factory with openai client.

        Args:
            model_id (Optional[str]): The ID of the OpenAI model to use. If None is provided, the first model in the list of models from the API backend will be used.
            tokenizer_name_or_path (Optional[str]): The name or path of the tokenizer to use. If None is given, it will use the tiktoken tokenizer from openai.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
            base_url (Optional[str], optional): The base URL for the OpenAI API. Defaults to None.
            api_engine (Optional[Literal[KNOWN_BACKEND]], optional): The backend LLM engine. This will help to decide the way of doing structured text generation. If not provided, not all structured text generation methods might work properly. Defaults to None.
            api_key (Optional[str], optional): The API key for the OpenAI API. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the BaseEngine initializer.
        """
        from .Engine.openai_engine import OpenAIEngine
        engine = OpenAIEngine(
            model_id=model_id,
            tokenizer_name_or_path=tokenizer_name_or_path,
            tokenizer_kwargs=tokenizer_kwargs,
            base_url=base_url,
            api_engine=api_engine,
            api_key=api_key,
            **kwargs
        )
        return cls(engine)
    
    @classmethod
    def from_huggingface(cls,
            pretrained_model_name_or_path: str,
            tokenizer_name_or_path: Optional[str] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> LLMFactory:
        """Initializes the HuggingFaceEngine with the given parameters.

        Args:
            pretrained_model_name_or_path (str): The HuggingFace repository name or the full path of the model file.
            tokenizer_name_or_path (str, optional): The Huggingface repository name or the full path to load the HuggingFace tokenizer. If not provided, pretrained_model_name_or_path would be used. Defaults to None.
            model_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the model. Defaults to None.
            tokenizer_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the BaseEngine class.
        """
        from .Engine.huggingface_engine import HuggingFaceEngine
        engine = HuggingFaceEngine(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs
        )
        return cls(engine)