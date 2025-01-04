Module llmflex.TextSplitter.markdown_splitter
=============================================

Classes
-------

`MarkdownTextSplitter(tokenizer: BaseTokenizer, chunk_size: int = 400, config: Dict[str, bool] | None = None)`
:   MarkdownTextSplitter is a text splitter that recursively splits text into smaller chunks of markdown sections.
        
    
    Initializes the MarkdownTextSplitter with a tokenizer, split strings, and chunk size.
    
    Args:
        tokenizer (BaseTokenizer): The tokenizer to use for counting tokens.
        chunk_size (int, optional): The maximum number of tokens allowed in each chunk. Defaults to 400.
        config (config: Optional[Dict[str, bool]], optional): The split tokens and if they should be prefix of chunks or not. If None is given, it will be split by headers. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.TextSplitter.base_splitter.BaseTextSplitter
    * abc.ABC