Module llmflex.Embeddings.api_embeddings
========================================

Classes
-------

`APIEmbeddings(base_url: str, encode_kwargs: Optional[Dict[str, Any]] = None)`
:   Base class for embeddings model.
        
    
    Initialising the embedding model instance.
    
    Args:
        base_url (str): URL for the api.
        encode_kwargs (Optional[Dict[str, Any]], optional): Encoding keyword arguments for the sentence transformer model. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Embeddings.base_embeddings.BaseEmbeddings
    * abc.ABC

`APIEmbeddingsToolkit(base_url: str, chunk_size: Optional[int] = None, chunk_overlap_perc: float = 0.1, encode_kwargs: Optional[Dict[str, Any]] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None)`
:   Base class for storing the embedding model and the text splitter.
        
    
    Initialising the self-hosted Huggingface API embeddings toolkit.
    
    Args:
        base_url (str): Model id (from Huggingface) to use.
        chunk_size (Optional[int], optional): Chunk size for the text splitter. If not provided, the min of the model max_seq_length or 512 will be used. Defaults to None.
        chunk_overlap_perc (float, optional): Number of tokens percentage overlap during text splitting. Defaults to 0.1.
        encode_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for encoding text. If None is given, the default is normalize_embeddings=True, batch_size=128. Defaults to None.
        tokenizer_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for loading tokenizer. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit