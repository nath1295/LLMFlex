Module llmflex.Embeddings.huggingface_embeddings
================================================

Classes
-------

`HuggingFaceEmbeddings(model_id: str, model_kwargs: Optional[Dict[str, Any]] = None, encode_kwargs: Optional[Dict[str, Any]] = None)`
:   Embeddings model from HuggingFace using sentence transformers.
    
    Initialising the embedding model.
    
    Args:
        model_id (str): Huggingface repo ID.
        model_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for loading the model. Defaults to None.
        encode_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for encoding text. If None is given, the default is normalize_embeddings=True, batch_size=128. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Embeddings.base_embeddings.BaseEmbeddings
    * abc.ABC

`HuggingfaceEmbeddingsToolkit(model_id: str, chunk_size: Optional[int] = None, chunk_overlap_perc: float = 0.1, model_kwargs: Optional[Dict[str, Any]] = None, encode_kwargs: Optional[Dict[str, Any]] = None)`
:   Base class for storing the embedding model and the text splitter.
        
    
    Initialising the Huggingface embeddings toolkit.
    
    Args:
        model_id (str): Model id (from Huggingface) to use.
        chunk_size (Optional[int], optional): Chunk size for the text splitter. If not provided, the min of the model max_seq_length or 512 will be used. Defaults to None.
        chunk_overlap_perc (float, optional): Number of tokens percentage overlap during text splitting. Defaults to 0.1.
        model_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for loading the model. Defaults to None.
        encode_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for encoding text. If None is given, the default is normalize_embeddings=True, batch_size=128. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit