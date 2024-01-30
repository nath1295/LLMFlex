Module llmplus.Models.Cores.exllamav2_core
==========================================

Functions
---------

    
`get_exl2_model_dir(repo_id: str, revision: Optional[str] = None) ‑> str`
:   

Classes
-------

`Exl2Core(repo_id: str, revision: Optional[str] = None, **kwargs)`
:   Base class of Core object to store the llm model and tokenizer.
        
    
    Initialising the core instance.
    
    Args:
        model_id (str, optional): Model id (from Huggingface) to use. Defaults to 'gpt2'.

    ### Ancestors (in MRO)

    * llmplus.Models.Cores.base_core.BaseCore

    ### Methods

    `decode(self, token_ids: List[int]) ‑> str`
    :   Untokenize a list of tokens.
        
        Args:
            token_ids (List[int]): Token ids to untokenize.
        
        Returns:
            str: Untokenized string.

`Exl2LLM(core: llmplus.Models.Cores.exllamav2_core.Exl2Core, temperature: float = 0, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True)`
:   Base LLM class for llmplus, using the LLM class from langchain.
        
    
    Initialising the llm.
    
    Args:
        core (Exl2Core): The Exl2Core core.
        temperature (float, optional): Set how "creative" the model is, the smaller it is, the more static of the output. Defaults to 0.
        max_new_tokens (int, optional): Maximum number of tokens to generate by the llm. Defaults to 2048.
        top_p (float, optional): While sampling the next token, only consider the tokens above this p value. Defaults to 0.95.
        top_k (int, optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to 40.
        repetition_penalty (float, optional): The value to penalise the model for generating repetitive text. Defaults to 1.1.
        stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. Defaults to None.
        stop_newline_version (bool, optional): Whether to add duplicates of the list of stop words starting with a new line character. Defaults to True.

    ### Ancestors (in MRO)

    * llmplus.Models.Cores.base_core.BaseLLM
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

    `core: llmplus.Models.Cores.exllamav2_core.Exl2Core`
    :

    `generation_config: Dict[str, Any]`
    :

    `stop: List[str]`
    :