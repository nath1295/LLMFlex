Module llmplus.Models.Cores.huggingface_core
============================================

Classes
-------

`HuggingfaceCore(model_id: str, model_type: Literal['default', 'awq', 'gptq'], model_kwargs: Dict[str, Any] = {}, tokenizer_kwargs: Dict[str, Any] = {})`
:   This is the core class of loading model in awq, gptq, or original format.
        
    
    Initiating the core with transformers.
    
    Args:
        model_id (str): Model id (from Huggingface) to use.
        model_type (Literal[&#39;default&#39;, &#39;awq&#39;, &#39;gptq&#39;]): Type of model format.
        model_kwargs (Dict[str, Any], optional): Keyword arguments for loading the model. Defaults to dict().
        tokenizer_kwargs (Dict[str, Any], optional): Keyword arguments for loading the tokenizer. Defaults to dict().

    ### Ancestors (in MRO)

    * llmplus.Models.Cores.base_core.BaseCore

    ### Instance variables

    `model_type: str`
    :   Format of the model.
        
        Returns:
            str: Format of the model.

`HuggingfaceLLM(core: llmplus.Models.Cores.huggingface_core.HuggingfaceCore, temperature: float = 0, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True)`
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

    `core: llmplus.Models.Cores.huggingface_core.HuggingfaceCore`
    :

    `generation_config: Dict[str, Any]`
    :

    `stop: List[str]`
    :

`KeywordsStoppingCriteria(stop_words: List[str], tokenizer: Any)`
:   class for handling stop words in transformers.pipeline

    ### Ancestors (in MRO)

    * transformers.generation.stopping_criteria.StoppingCriteria
    * abc.ABC

    ### Methods

    `get_min_ids(self, word: str) ‑> List[int]`
    :