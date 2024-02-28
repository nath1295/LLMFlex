Module llmflex.Models.Cores.utils
=================================

Functions
---------

    
`add_newline_char_to_stopwords(stop: List[str]) ‑> List[str]`
:   Create a duplicate of the stop words and add a new line character as a prefix to each of them if their prefixes are not new line characters.
    
    Args:
        stop (List[str]): List of stop words.
    
    Returns:
        List[str]: New version of the list of stop words, with new line characters.

    
`detect_prompt_template_by_id(model_id: str) ‑> str`
:   Guess the prompt format for the model by model ID.
    
    Args:
        model_id (str): Huggingface ID of the model.
    
    Returns:
        str: Prompt template preset.

    
`detect_prompt_template_by_jinja(jinja_template: str) ‑> str`
:   Detect if the jinja template given is the same as one of the presets.
    
    Args:
        jinja_template (str): Jinja template to test.
    
    Returns:
        str: Prompt template preset.

    
`find_roots(text: str, stop: List[str], stop_len: List[int]) ‑> Tuple[str, str]`
:   This function is a helper function for stopping stop words from showing up while doing work streaming in some custom llm classes. Not intended to be used alone.
    
    Args:
        text (str): Output of the model.
        stop (List[str]): List of stop words.
        stop_len (List[int]): List of the lengths of the stop words.
    
    Returns:
        Tuple[str, str]: Curated output of the model, potential root of stop words.

    
`get_prompt_template_by_jinja(model_id: str, tokenizer: Any) ‑> llmflex.Prompts.prompt_template.PromptTemplate`
:   Getting the appropriate prompt template given the huggingface tokenizer.
    
    Args:
        model_id (str): Repo ID of the tokenizer.
        tokenizer (Any): Huggingface tokenizer.
    
    Returns:
        PromptTemplate: The prompt template object.

    
`get_stop_words(stop: Optional[List[str]], tokenizer: Any, add_newline_version: bool = True, tokenizer_type: Literal['transformers', 'llamacpp', 'openai'] = 'transformers') ‑> List[str]`
:   Adding necessary stop words such as EOS token and multiple newline characters.
    
    Args:
        stop (Optional[List[str]]): List of stop words, if None is given, an empty list will be assumed.
        tokenizer (Any): Tokenizer to get the EOS token.
        add_newline_version (bool, optional): Whether to use add_newline_char_to_stopwords function. Defaults to True.
        tokenizer_type (Literal[&#39;transformers&#39;, &#39;llamacpp&#39;, &#39;openai&#39;], optional): Type of tokenizer. Defaults to 'transformers'.
    
    Returns:
        List[str]: Updated list of stop words.

    
`list_local_models() ‑> List[Dict[str, str]]`
:   Check what you have in your local model cache directory.
    
    Returns:
        List[Dict[str, str]]: List of dictionarys of model details.

    
`textgen_iterator(text_generator: Iterator[str], stop: List[str]) ‑> Iterator[str]`
:   Make a text generator stop before spitting out the stop words.
    
    Args:
        text_generator (Iterator[str]): Text generator to transform.
        stop (List[str]): Stop words.
    
    Yields:
        Iterator[str]: Text generator with stop words applied.