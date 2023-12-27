Module llmplus.Models.Cores.llamacpp_core
=========================================

Classes
-------

`LlamaCppCore(model_id: str, model_file: Optional[str] = None, context_length: int = 4096, **kwargs)`
:   This is the core class of loading model in gguf format.
        
    
    Initialising the core.
    
    Args:
        model_id (str): Model id (from Huggingface) to use.
        model_file (Optional[str], optional): Specific GGUF model to use. If None, the lowest quant will be used. Defaults to None.
        context_length (int, optional): Context length of the model. Defaults to 4096.

    ### Ancestors (in MRO)

    * llmplus.Models.Cores.base_core.BaseCore

`LlamaCppLLM(core: llmplus.Models.Cores.llamacpp_core.LlamaCppCore, temperature: float = 0, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True)`
:   Custom implementation of streaming for models loaded with `llama-cpp-python`, Used in the Llm factory to get new llm from the model.
    
    Initialising the llm.
    
    Args:
        core (LlamaCppCore): The LlamaCppCore core.
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

    `core: llmplus.Models.Cores.llamacpp_core.LlamaCppCore`
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