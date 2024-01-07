Module llmplus.Models.Cores.utils
=================================

Functions
---------

    
`add_newline_char_to_stopwords(stop: List[str]) ‑> List[str]`
:   Create a duplicate of the stop words and add a new line character as a prefix to each of them if their prefixes are not new line characters.
    
    Args:
        stop (List[str]): List of stop words.
    
    Returns:
        List[str]: New version of the list of stop words, with new line characters.

    
`find_roots(text: str, stop: List[str], stop_len: List[int]) ‑> Tuple[str, str]`
:   This function is a helper function for stopping stop words from showing up while doing work streaming in some custom llm classes. Not intended to be used alone.
    
    Args:
        text (str): Output of the model.
        stop (List[str]): List of stop words.
        stop_len (List[int]): List of the lengths of the stop words.
    
    Returns:
        Tuple[str, str]: Curated output of the model, potential root of stop words.

    
`get_stop_words(stop: Optional[List[str]], tokenizer: Any, add_newline_version: bool = True, tokenizer_type: Literal['transformers', 'llamacpp', 'openai'] = 'transformers') ‑> List[str]`
:   Adding necessary stop words such as EOS token and multiple newline characters.
    
    Args:
        stop (Optional[List[str]]): List of stop words, if None is given, an empty list will be assumed.
        tokenizer (Any): Tokenizer to get the EOS token.
        add_newline_version (bool, optional): Whether to use add_newline_char_to_stopwords function. Defaults to True.
        tokenizer_type (Literal[&#39;transformers&#39;, &#39;llamacpp&#39;, &#39;openai&#39;], optional): Type of tokenizer. Defaults to 'transformers'.
    
    Returns:
        List[str]: Updated list of stop words.

    
`textgen_iterator(text_generator: Iterator[str], stop: List[str]) ‑> Iterator[str]`
:   Make a text generator stop before spitting out the stop words.
    
    Args:
        text_generator (Iterator[str]): Text generator to transform.
        stop (List[str]): Stop words.
    
    Yields:
        Iterator[str]: Text generator with stop words applied.