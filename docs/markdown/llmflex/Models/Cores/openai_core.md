Module llmflex.Models.Cores.openai_core
=======================================

Functions
---------

    
`parse_sse(response: Response) ‑> Iterator[Dict[str, Any]]`
:   Parsing streaming response from llama server.
    
    Args:
        response (Response): Response object from the llama server.
    
    Yields:
        Generator[Dict[str, Any]]: Dictionary with text tokens in the content.

Classes
-------

`OpenAICore(base_url: Optional[str] = None, api_key: Optional[str] = None, model_id: Optional[str] = None, tokenizer_id: Optional[str] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None)`
:   Core class for llm models using openai api interface.
        
    
    Initialising the llm core.
    
    Args:
        base_url (Optional[str], optional): URL for the model api endpoint, if None is given, it will use the default URL for OpenAI api. Defaults to None.
        api_key (Optional[str], optional): If using OpenAI api, API key should be provided. Defaults to None.
        model_id (Optional[str], optional): If using OpenAI api or using an api with multiple models, please provide the model to use. Otherwise 'gpt-3.5-turbo' or the first available model will be used by default. Defaults to None.
        tokenizer_id (Optional[str], optional): If not using OpenAI api, repo_id to get the tokenizer from HuggingFace must be provided. Defaults to None.
        tokenizer_kwargs (Optional[Dict[str, Any]], optional): If not using OpenAI api, kwargs can be passed to load the tokenizer from HuggingFace. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Models.Cores.base_core.BaseCore
    * abc.ABC

    ### Static methods

    `from_model_object(model: Any, tokenizer: Any, model_id: Optional[str] = None, **kwargs) ‑> llmflex.Models.Cores.openai_core.OpenAICore`
    :   Load a core directly from an already loaded model object and a tokenizer object for the supported formats.
        
        Args:
            model (Any): The model object.
            tokenizer (Any): The tokenizer object.
            model_id (Optional[str], optional): The model_id. Defaults to None.
        
        Returns:
            OpenAICore: The initialised core.

    ### Instance variables

    `base_url: str`
    :   The base url of the API.
        
        Returns:
            str: The base url of the API.

    `is_llama: bool`
    :   Whether or not the server is a llama.cpp server.
        
        Returns:
            bool: Whether or not the server is a llama.cpp server.

    ### Methods

    `llama_generate(self, prompt: str, temperature: float = 0, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True, stream: bool = False, **kwargs) ‑> Union[str, Iterator[str]]`
    :   Generate the output with the given prompt for llama.cpp server.
        
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