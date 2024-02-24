Module llmplus.Embeddings.api_embeddings
========================================

Classes
-------

`APIEmbeddings(base_url: str, encode_kwargs: Dict[str, Any] = {})`
:   Interface for embedding models.

    ### Ancestors (in MRO)

    * langchain_core.embeddings.Embeddings
    * abc.ABC

    ### Methods

    `embed_documents(self, texts: List[str]) ‑> List[List[float]]`
    :   embed search docs.
        
        Args:
            texts (List[str]): List of texts to embed.
        
        Returns:
            List[List[float]]: List of embeddings given the texts.

    `embed_query(self, text: str) ‑> List[float]`
    :   Embed query text.
        
        Args:
            text (str): Text to embed.
        
        Returns:
            List[float]: Embeddings of the text.

`APIEmbeddingsToolkit(base_url: str, chunk_size: Optional[int] = None, chunk_overlap_perc: float = 0.1, encode_kwargs: Dict[str, Any] = {'normalize_embeddings': True, 'batch_size': 128}, tokenizer_kwargs: Dict[str, Any] = {})`
:   Base class for storing the embedding model and the text splitter.
        
    
    Initialising the self-hosted Huggingface API embeddings toolkit.
    
    Args:
        base_url (str): Model id (from Huggingface) to use.
        chunk_size (Optional[int], optional): Chunk size for the text splitter. If not provided, the min of the model max_seq_length or 512 will be used. Defaults to None.
        chunk_overlap_perc (float, optional): Number of tokens percentage overlap during text splitting. Defaults to 0.1.
        encode_kwargs (Dict[str, Any], optional): Keyword arguments for encoding. Defaults to dict(normalize_embeddings=True).
        tokenizer_kwargs (Dict[str, Any], optional): Keyword arguments for loading tokenizer. Defaults to dict().

    ### Ancestors (in MRO)

    * llmplus.Embeddings.base_embeddings.BaseEmbeddingsToolkit