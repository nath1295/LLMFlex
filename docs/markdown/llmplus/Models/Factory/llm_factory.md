Module llmplus.Models.Factory.llm_factory
=========================================

Functions
---------

    
`detect_model_type(model_id: str) ‑> str`
:   This function attempts to get the model format type with the model id.
    
    Args:
        model_id (str): Model ID form Huggingface.
    
    Returns:
        str: Model format type.

Classes
-------

`LlmFactory(model_id: str, model_type: Literal['auto', 'default', 'gptq', 'awq', 'gguf', 'openai', 'exl2', 'debug'] = 'auto', model_file: Optional[str] = None, model_kwargs: Dict[str, Any] = {}, revision: Optional[str] = None, from_local: bool = False, context_length: int = 4096, base_url: Optional[str] = None, api_key: Optional[str] = None, tokenizer_id: Optional[str] = None, tokenizer_kwargs: Dict[str, Any] = {}, **kwargs)`
:   Initialise the model core to create LLMs.
    
    Args:
        model_id (str): Model ID (from Huggingface) to use or the model to use if using OpenAI API core.
        model_type (Literal[&#39;auto&#39;, &#39;default&#39;, &#39;gptq&#39;, &#39;awq&#39;, &#39;gguf&#39;, &#39;openai&#39;, &#39;exl2&#39;, &#39;debug&#39;], optional): Type of model format, if 'auto' is given, model_type will be automatically detected. Defaults to 'auto'.
        model_file (Optional[str], optional): Specific model file to use. Only useful for `model_type="gguf"`. Defaults to None.
        model_kwargs (Dict[str, Any], optional): Keyword arguments for loading the model. Only useful for Default, GPTQ, and AWQ models. Defaults to dict().
        revision (Optional[str], optional): Specific revision of the model repository. Only useful for `model_type="exl2"`. Defaults to None.
        from_local (bool, optional): Whether to treat the model_id given as a local path or a Huggingface ID. Only useful for GGUF models. Defaults to False.
        context_length (int, optional): Size of the context window. Only useful for GGUF models. Defaults to 4096.
        base_url (Optional[str], optional): Base URL for the API. Only useful for OpenAI APIs. Defaults to None.
        api_key (Optional[str], optional): API key for OpenAI API. Defaults to None.
        tokenizer_id (Optional[str], optional): Model ID (from Huggingface) to load the tokenizer. Useful for model types "default", "gptq", "awq", and "openai". Defaults to None.
        tokenizer_kwargs (Dict[str, Any], optional): Keyword arguments for loading the tokenizer. Useful for model types "default", "gptq", "awq", and "openai".  Defaults to dict().

    ### Instance variables

    `core: Type[llmplus.Models.Cores.base_core.BaseCore]`
    :   Core model of the llm factory.
        
        Returns:
            Type[BaseCore]: Core model of the llm factory.

    `model_id: str`
    :   Model id (from Huggingface).
        
        Returns:
            str: Model id (from Huggingface).

    `model_type: str`
    :   Type of model format.
        
        Returns:
            str: Type of model format.

    `prompt_template: llmplus.Prompts.prompt_template.PromptTemplate`
    :   Default prompt template for the model.
        
        Returns:
            PromptTemplate: Default prompt template for the model.

    ### Methods

    `call(self, temperature: float = 0.8, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, newline=True, **kwargs: Dict[str, Any]) ‑> Type[llmplus.Models.Cores.base_core.BaseLLM]`
    :   Calling the object will create a langchain format llm with the generation configurations passed from the arguments. 
        
        Args:
            temperature (float, optional): Set how "creative" the model is, the samller it is, the more static of the output. Defaults to 0.8.
            max_new_tokens (int, optional): Maximum number of tokens to generate by the llm. Defaults to 2048.
            top_p (float, optional): While sampling the next token, only consider the tokens above this p value. Defaults to 0.95.
            top_k (int, optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to 40.
            repetition_penalty (float, optional): The value to penalise the model for generating repetitive text. Defaults to 1.1.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. Defaults to None.
            newline (bool, optional): Whether to add a newline character to the beginning of the "stop" list provided. Defaults to True.
        
        Returns:
            BaseLLM: An LLM.