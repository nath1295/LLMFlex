Module llmplus.Models.Cores.llamacpp_core
=========================================

Functions
---------

    
`get_model_dir(model_id: str, model_file: Optional[str] = None) ‑> str`
:   Download the model file from Huggingface and get the local directory.
    
    Args:
        model_id (str): Model's HuggingFace ID.
        model_file (Optional[str], optional): Specific model quant file. If None, will choose the smallest quant automatically. Defaults to None.
    
    Returns:
        str: Local directory of the model file.

Classes
-------

`LlamaCppCore(model_id_or_path: str, model_file: Optional[str] = None, context_length: int = 4096, from_local: bool = False, **kwargs)`
:   This is the core class of loading model in gguf format.
        
    
    Initialising the core.
    
    Args:
        model_id (str): Model id (from Huggingface) or model file path to use.
        model_file (Optional[str], optional): Specific GGUF model to use. If None, the lowest quant will be used. Defaults to None.
        context_length (int, optional): Context length of the model. Defaults to 4096.
        from_local (bool, optional): Whether to treat the model_id given as a local path or a Huggingface ID. Defaults to False.

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

    `core: llmplus.Models.Cores.llamacpp_core.LlamaCppCore`
    :

    `generation_config: Dict[str, Any]`
    :

    `stop: List[str]`
    :