Module llmflex.LLM.llm_factory
==============================

Classes
-------

`LLMFactory(engine: BaseEngine)`
:   Class for creating LLMs with different default generation settings from the same LLM engine.
        
    
    Initialises the class with the underlying LLM engine.
    
    Args:
        engine (BaseEngine): The underlying LLM engine.

    ### Static methods

    `from_huggingface(pretrained_model_name_or_path: str, tokenizer_name_or_path: Optional[str] = None, model_kwargs: Optional[Dict[str, Any]] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None, **kwargs) ‑> llmflex.LLM.llm_factory.LLMFactory`
    :   Initializes the HuggingFaceEngine with the given parameters.
        
        Args:
            pretrained_model_name_or_path (str): The HuggingFace repository name or the full path of the model file.
            tokenizer_name_or_path (str, optional): The Huggingface repository name or the full path to load the HuggingFace tokenizer. If not provided, pretrained_model_name_or_path would be used. Defaults to None.
            model_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the model. Defaults to None.
            tokenizer_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the BaseEngine class.

    `from_llamacpp(pretrained_model_name_or_path: str, model_file: Optional[str] = None, tokenizer_name_or_path: Optional[str] = None, context_size: int = 8192, verbose: bool = False, download_kwargs: Optional[Dict[str, Any]] = None, **kwargs) ‑> llmflex.LLM.llm_factory.LLMFactory`
    :   Initializes the llm factory with llama-cpp-python.
        
        Args:
            pretrained_model_name_or_path (str): The HuggingFace repository name or the full path of the model file.
            model_file (str, optional): The model filename if a HuggingFace repository name is given for tokenizer_id_or_path. Defaults to None.
            tokenizer_name_or_path (str, optional): The Huggingface repository name or the full path to load the HuggingFace tokenizer. If not provided, structured text generation might encounter problems. Defaults to None.
            context_size (int, optional): The size of the context window. Defaults to 8192.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            download_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the download function. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the Llama class.

    `from_mlx(pretrained_model_name_or_path: str, quantization: "Literal['fp16', 'q8', 'q4', 'q2']" = 'fp16', tokenizer_name_or_path: Optional[str] = None, revision: Optional[str] = None, model_kwargs: Optional[Dict[str, Any]] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None, prefill_step_size: int = 512, **kwargs) ‑> llmflex.LLM.llm_factory.LLMFactory`
    :   Initializes the llm factory with mlx-textgen.
        
        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained model to use.
            quantization (Literal['fp16', 'q8', 'q4', 'q2'], optional): The quantization method to use for the model. Defaults to 'fp16'.
            tokenizer_name_or_path (Optional[str], optional): The name or path of the tokenizer to use. Defaults to None.
            revision (Optional[str], optional): The branch of the Huggingface repository to use. Defaults to None.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
            prefill_step_size (int, optional): The batch size of prompt processing. Defaults to 512.

    `from_openai(model_id: Optional[str], tokenizer_name_or_path: Optional[str], tokenizer_kwargs: Optional[Dict[str, Any]] = None, base_url: Optional[str] = None, api_engine: "Optional[Literal['openai', 'mlx-textgen', 'vllm', 'llama.cpp', 'llama-cpp-python']]" = None, api_key: Optional[str] = None, **kwargs) ‑> llmflex.LLM.llm_factory.LLMFactory`
    :   Initializes the llm factory with openai client.
        
        Args:
            model_id (Optional[str]): The ID of the OpenAI model to use. If None is provided, the first model in the list of models from the API backend will be used.
            tokenizer_name_or_path (Optional[str]): The name or path of the tokenizer to use. If None is given, it will use the tiktoken tokenizer from openai.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
            base_url (Optional[str], optional): The base URL for the OpenAI API. Defaults to None.
            api_engine (Optional[Literal[KNOWN_BACKEND]], optional): The backend LLM engine. This will help to decide the way of doing structured text generation. If not provided, not all structured text generation methods might work properly. Defaults to None.
            api_key (Optional[str], optional): The API key for the OpenAI API. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the BaseEngine initializer.

    ### Instance variables

    `chat_template: ChatTemplate`
    :   Returns the default chat template for the LLM.
        
        Returns:
            ChatTemplate: The default chat template for the LLM.

    `engine: BaseEngine`
    :   Engine used in the LLMs.
        
        Returns:
            BaseEngine: Engine used in the LLMs.

    `engine_class: str`
    :   The class name of the LLM engine.
        
        Returns:
            str: The class name of the LLM engine.

    `model_id: str`
    :   Returns the ID of the LLM model.
        
        Returns:
            str: The ID of the LLM model.

    `model_name: str`
    :   Returns the name of the LLM model.
        
        Returns:
            str: The name of the LLM model.

    `tokenizer: BaseTokenizer`
    :   Returns the tokenizer used in the LLM.
        
        Returns:
            BaseTokenizer: The tokenizer used in the LLM.