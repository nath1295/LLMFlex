Module llmflex.LLM.Engine.huggingface_engine
============================================

Functions
---------

`sequence_overlap(s1: str, s2: str) ‑> bool`
:   Determine if there is overlapping string between the suffix of the first string and the prefix of the second string.
    
    Args:
        s1 (str): First string.
        s2 (str): Second string.
    
    Returns:
        bool: Whether there is overlap.

`stopping_criteria(text: str, stop_tuple: List[Tuple[str, int]]) ‑> llmflex.LLM.Engine.huggingface_engine.StopCondition`
:   Get the stopping condition with stop words and eos token.
    
    Args:
        text (str): Text to be determined to stop or not.
        stop_tuple (List[Tuple[str, int]]): List of tuple with the stop word string and the length of the string. Must be ordered descendingly by length.
    
    Returns:
        StopCondition: A status class stating whether the generation should stop and the length of text to trim.

Classes
-------

`HuggingFaceEngine(pretrained_model_name_or_path: str, tokenizer_name_or_path: str | None = None, model_kwargs: Dict[str, Any] | None = None, tokenizer_kwargs: Dict[str, Any] | None = None, **kwargs)`
:   Class for LLM engine using llama-cpp-python.
        
    
    Initializes the HuggingFaceEngine with the given parameters.
    
    Args:
        pretrained_model_name_or_path (str): The HuggingFace repository name or the full path of the model file.
        tokenizer_name_or_path (str, optional): The Huggingface repository name or the full path to load the HuggingFace tokenizer. If not provided, pretrained_model_name_or_path would be used. Defaults to None.
        model_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the model. Defaults to None.
        tokenizer_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the BaseEngine class.

    ### Ancestors (in MRO)

    * llmflex.LLM.Engine.base_engine.BaseEngine
    * abc.ABC

`StopCondition(stop_met: bool, trim_length: int, stop_text: str | None = None)`
:   StopCondition(stop_met, trim_length, stop_text)

    ### Ancestors (in MRO)

    * builtins.tuple

    ### Instance variables

    `stop_met: bool`
    :   Alias for field number 0

    `stop_text: str | None`
    :   Alias for field number 2

    `trim_length: int`
    :   Alias for field number 1

`TokenStreamer(pretrained_model_name_or_path: str, tokenizer: transformers.tokenization_utils.PreTrainedTokenizer)`
:   Class for streaming text tokens.

    ### Ancestors (in MRO)

    * transformers.generation.streamers.BaseStreamer

    ### Methods

    `end(self)`
    :   Function that is called by `.generate()` to signal the end of generation

    `put(self, value)`
    :   Function that is called by `.generate()` to push new tokens