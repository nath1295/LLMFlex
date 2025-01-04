Module llmflex.TextSplitter.semantic_splitter
=============================================

Classes
-------

`SemanticTextSplitter(embeddings: BaseEmbeddings, tokenizer: ForwardRef('BaseTokenizer') | None = None, chunk_size: int = 400, split_fn: Callable[[str], List[str]] | None = None)`
:   A text splitter that splits text into chunks based on semantic similarity.
    
    This splitter uses embeddings to determine the semantic similarity between chunks of text.
    It splits the text into chunks such that each chunk is as semantically similar as possible to the others.
    
    Initializes the SemanticTextSplitter.
    
    Args:
        embeddings (BaseEmbeddings): The embeddings model to use for semantic similarity.
        tokenizer (Optional[BaseTokenizer], optional): The tokenizer to use for counting tokens. If None is given, the tokenizer of the embedding model will be used. Defaults to None.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 400.
        split_fn (Optional[Callable[[str], List[str]]], optional): A custom function to split the text, If None is given, a sentence text splitter will be used with chunk size being one third of the SemanticTextSplitter chunk size. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.TextSplitter.base_splitter.BaseTextSplitter
    * abc.ABC