Module llmplus.Models.Cores.openai_core
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

    * llmplus.Models.Cores.base_core.BaseCore

`OpenAILLM(core: llmplus.Models.Cores.openai_core.OpenAICore, temperature: float = 0, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True)`
:   Custom implementation of streaming for models from OpenAI api. Used in the Llm factory to get new llm from the model.
    
    Initialising the llm.
    
    Args:
        core (OpenAICor): The OpenAICore core.
        temperature (float, optional): Set how "creative" the model is, the smaller it is, the more static of the output. Defaults to 0.
        max_new_tokens (int, optional): Maximum number of tokens to generate by the llm. Defaults to 2048.
        top_p (float, optional): While sampling the next token, only consider the tokens above this p value. Defaults to 0.95.
        top_k (int, optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to 40.
        repetition_penalty (float, optional): The value to penalise the model for generating repetitive text. Defaults to 1.1.
        stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. Defaults to None.
        stop_newline_version (bool, optional): Whether to add duplicates of the list of stop words starting with a new line character. Defaults to True.

    ### Ancestors (in MRO)

    * langchain_core.language_models.llms.LLM
    * langchain_core.language_models.llms.BaseLLM
    * langchain_core.language_models.base.BaseLanguageModel
    * langchain_core.runnables.base.RunnableSerializable
    * langchain_core.load.serializable.Serializable
    * pydantic.v1.main.BaseModel
    * pydantic.v1.utils.Representation
    * langchain_core.runnables.base.Runnable
    * typing.Generic
    * abc.ABC

    ### Class variables

    `core: llmplus.Models.Cores.openai_core.OpenAICore`
    :

    `generation_config: Dict[str, Any]`
    :

    `stop: List[str]`
    :

    ### Methods

    `get_num_tokens(self, text: str) ‑> int`
    :   Get the number of tokens given the text string.
        
        Args:
            text (str): Text
        
        Returns:
            int: Number of tokens

    `get_token_ids(self, text: str) ‑> List[int]`
    :   Get the token ids of the given text.
        
        Args:
            text (str): Text
        
        Returns:
            List[int]: List of token ids.

    `stream(self, input: str, config: Optional[langchain_core.runnables.config.RunnableConfig] = None, *, stop: Optional[List[str]] = None, **kwargs) ‑> Iterator[str]`
    :   Text streaming of llm generation. Return a python generator of output tokens of the llm given the prompt.
        
        Args:
            input (str): The prompt to the llm.
            config (Optional[RunnableConfig]): Not used. Defaults to None.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. If provided, it will overide the original llm stop list. Defaults to None.
        
        Yields:
            Iterator[str]: The next generated token.