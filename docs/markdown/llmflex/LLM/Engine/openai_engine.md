Module llmflex.LLM.Engine.openai_engine
=======================================

Classes
-------

`OpenAIEngine(model_id: str | None, tokenizer_name_or_path: str | None, tokenizer_kwargs: Dict[str, Any] | None = None, base_url: str | None = None, api_engine: Literal['openai', 'mlx-textgen', 'vllm', 'llama.cpp', 'llama-cpp-python'] | None = None, api_key: str | None = None, **kwargs)`
:   Class for LLM engine using openai API.
        
    
    Initializes the OpenAIEngine with the given parameters.
    
    Args:
        model_id (str): The ID of the OpenAI model to use. If None is provided, the first model in the list of models from the API backend will be used.
        tokenizer_name_or_path (Optional[str]): The name or path of the tokenizer to use. If None is given, it will use the tiktoken tokenizer from openai.
        tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
        base_url (Optional[str], optional): The base URL for the OpenAI API. Defaults to None.
        api_engine (Optional[KNOWN_BACKEND], optional): The backend LLM engine. This will help to decide the way of doing structured text generation. If not provided, not all structured text generation methods might work properly. Defaults to None.
        api_key (Optional[str], optional): The API key for the OpenAI API. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the BaseEngine initializer.

    ### Ancestors (in MRO)

    * llmflex.LLM.Engine.base_engine.BaseEngine
    * abc.ABC