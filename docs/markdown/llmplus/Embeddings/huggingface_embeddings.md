Module llmplus.Embeddings.huggingface_embeddings
================================================

Classes
-------

`HuggingfaceEmbeddingsToolkit(model_id: str, chunk_size: Optional[int] = None, chunk_overlap_perc: float = 0.1, model_kwargs: Dict[str, Any] = {}, encode_kwargs: Dict[str, Any] = {'normalize_embeddings': True, 'batch_size': 128})`
:   Base class for storing the embedding model and the text splitter.
        
    
    Initialising the Huggingface embeddings toolkit.
    
    Args:
        model_id (str): Model id (from Huggingface) to use.
        chunk_size (Optional[int], optional): Chunk size for the text splitter. If not provided, the min of the model max_seq_length or 512 will be used. Defaults to None.
        chunk_overlap_perc (float, optional): Number of tokens percentage overlap during text splitting. Defaults to 0.1.
        model_kwargs (Dict[str, Any], optional): Keyword arguments for the model. Defaults to dict().
        encode_kwargs (Dict[str, Any], optional): Keyword arguments for encoding. Defaults to dict(normalize_embeddings=True).

    ### Ancestors (in MRO)

    * llmplus.Embeddings.base_embeddings.BaseEmbeddingsToolkit