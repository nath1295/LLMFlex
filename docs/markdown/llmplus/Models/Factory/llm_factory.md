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

`LlmFactory(model_id: str, model_type: Literal['auto', 'default', 'gptq', 'awq', 'gguf', 'debug'] = 'auto', **kwargs)`
:   Initiate the model core to create LLMs.
    
    Args:
        model_id (str): Model id (from Huggingface) to use.
        model_type (Literal[&#39;auto&#39;, &#39;default&#39;, &#39;gptq&#39;, &#39;awq&#39;, &#39;gguf&#39;, &#39;debug&#39;], optional): Type of model format, if 'auto' is given, model_type will be automatically detected. Defaults to 'auto'.

    ### Instance variables

    `core: llmplus.Models.Cores.base_core.BaseCore`
    :   Core model of the llm factory.
        
        Returns:
            BaseCore: Core model of the llm factory.

    `model_id: str`
    :   Model id (from Huggingface).
        
        Returns:
            str: Model id (from Huggingface).

    `model_type: str`
    :   Type of model format.
        
        Returns:
            str: Type of model format.

    ### Methods

    `call(self, temperature: float = 0.8, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, newline=True, **kwargs: Dict[str, Any]) ‑> llmplus.Models.Cores.base_core.BaseLLM`
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