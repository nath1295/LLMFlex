Module llmflex.TextSplitter.recursive_splitter
==============================================

Classes
-------

`RecursiveTextSplitter(tokenizer: BaseTokenizer, split_strings: List[str] | None = None, chunk_size: int = 400)`
:   RecursiveTextSplitter is a text splitter that recursively splits text into smaller chunks until each chunk is below a specified maximum token length.
        
    
    Initializes the RecursiveTextSplitter with a tokenizer, split strings, and chunk size.
    
    Args:
        tokenizer (BaseTokenizer): The tokenizer to use for counting tokens.
        split_strings (Optional[List[str]], optional): A list of strings to split the text. If None is given, it will be ["\n\n", "\n", "?", "!", ".", ","]. Defaults to None.
        chunk_size (int, optional): The maximum number of tokens allowed in each chunk. Defaults to 400.

    ### Ancestors (in MRO)

    * llmflex.TextSplitter.base_splitter.BaseTextSplitter
    * abc.ABC