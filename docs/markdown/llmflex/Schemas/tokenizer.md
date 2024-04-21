Module llmflex.Schemas.tokenizer
================================

Classes
-------

`Tokenizer(tokenize_fn: Callable[[str], List[int]], detokenize_fn: Callable[[List[int]], str])`
:   Class to tokenize and detokenize strings.
        
    
    Initialising the tokenizer.
    
    Args:
        tokenize_fn (Callable[[str], List[int]]): Function to tokenize text.
        detokenize_fn (Callable[[List[int]], str]): Function to detokenize token ids.

    ### Methods

    `detokenize(self, token_ids: List[int]) ‑> str`
    :   Detokenize the given list of token ids.
        
        Args:
            token_ids (List[int]): List of token ids.
        
        Returns:
            str: Detokenized string.

    `get_num_tokens(self, text: str) ‑> int`
    :   Get the number of tokens in the given string.
        
        Args:
            text (str): String to count.
        
        Returns:
            int: Number of tokens in the given string.

    `split_text_on_tokens(self, text: str, chunk_size: int = 400, chunk_overlap: int = 40) ‑> List[str]`
    :   Split text base on token numbers.
        
        Args:
            text (str): Text to split.
            chunk_size (int, optional): Maximum number of tokens for each chunk. Defaults to 400.
            chunk_overlap (int, optional): Overlapping number of tokens for each consecutive chunk. Defaults to 40.
        
        Returns:
            List[str]: List of text chunks.

    `tokenize(self, text: str) ‑> List[str]`
    :   Tokenize the given string.
        
        Args:
            text (str): String to tokenize.
        
        Returns:
            List[str]: List of token ids.