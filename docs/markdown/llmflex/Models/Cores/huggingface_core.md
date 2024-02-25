Module llmflex.Models.Cores.huggingface_core
============================================

Classes
-------

`HuggingfaceCore(model_id: str, model_type: "Literal['default', 'awq', 'gptq']", model_kwargs: Dict[str, Any] = {}, tokenizer_kwargs: Dict[str, Any] = {})`
:   This is the core class of loading model in awq, gptq, or original format.
        
    
    Initiating the core with transformers.
    
    Args:
        model_id (str): Model id (from Huggingface) to use.
        model_type (Literal[&#39;default&#39;, &#39;awq&#39;, &#39;gptq&#39;]): Type of model format.
        model_kwargs (Dict[str, Any], optional): Keyword arguments for loading the model. Defaults to dict().
        tokenizer_kwargs (Dict[str, Any], optional): Keyword arguments for loading the tokenizer. Defaults to dict().

    ### Ancestors (in MRO)

    * llmflex.Models.Cores.base_core.BaseCore
    * abc.ABC

    ### Static methods

    `from_model_object(model: Any, tokenizer: Any, model_id: str = 'Unknown', model_type: "Literal['default', 'awq', 'gptq']" = 'default') ‑> llmflex.Models.Cores.huggingface_core.HuggingfaceCore`
    :   Load a core directly from an already loaded model object and a tokenizer object for the supported formats.
        
        Args:
            model (Any): The model object.
            tokenizer (Any): The tokenizer object.
            model_id (str): The model_id.
            model_type (Literal['default', 'awq', 'gptq']): The quantize type of the model.
        
        Returns:
            BaseCore: The initialised core.

    ### Instance variables

    `model_type: str`
    :   Format of the model.
        
        Returns:
            str: Format of the model.

`KeywordsStoppingCriteria(stop_words: List[str], tokenizer: Any)`
:   class for handling stop words in transformers.pipeline

    ### Ancestors (in MRO)

    * transformers.generation.stopping_criteria.StoppingCriteria
    * abc.ABC

    ### Methods

    `get_min_ids(self, word: str) ‑> List[int]`
    :