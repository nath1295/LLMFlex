Module llmflex.Models.Cores.openai_core
=======================================

Classes
-------

`OpenAICore(base_url: Optional[str] = None, api_key: Optional[str] = None, model_id: Optional[str] = None, tokenizer_id: Optional[str] = None, tokenizer_kwargs: Dict[str, Any] = {})`
:   Core class for llm models using openai api interface.
        
    
    Initialising the llm core.
    
    Args:
        base_url (Optional[str], optional): URL for the model api endpoint, if None is given, it will use the default URL for OpenAI api. Defaults to None.
        api_key (Optional[str], optional): If using OpenAI api, API key should be provided. Defaults to None.
        model_id (Optional[str], optional): If using OpenAI api or using an api with multiple models, please provide the model to use. Otherwise 'gpt-3.5-turbo' or the first available model will be used by default. Defaults to None.
        tokenizer_id (Optional[str], optional): If not using OpenAI api, repo_id to get the tokenizer from HuggingFace must be provided. Defaults to None.
        tokenizer_kwargs (Dict[str, Any], optional): If not using OpenAI api, kwargs can be passed to load the tokenizer from HuggingFace. Defaults to dict().

    ### Ancestors (in MRO)

    * llmflex.Models.Cores.base_core.BaseCore
    * abc.ABC