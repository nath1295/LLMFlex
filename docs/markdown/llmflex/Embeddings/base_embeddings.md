Module llmflex.Embeddings.base_embeddings
=========================================

Classes
-------

`BaseEmbeddings()`
:   Base class for embeddings model.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.Embeddings.api_embeddings.APIEmbeddings
    * llmflex.Embeddings.huggingface_embeddings.HuggingFaceEmbeddings

    ### Methods

    `embed_documents(self, texts: List[str]) ‑> List[List[float]]`
    :   Embed list of texts.
        
        Args:
            texts (List[str]): List of texts to embed.
        
        Returns:
            List[List[float]]: List of embedded vectors.

    `embed_query(self, text: str) ‑> List[float]`
    :   Embed one string.
        
        Args:
            text (str): String to embed.
        
        Returns:
            List[float]: embeddings of the string.

`BaseEmbeddingsToolkit(embedding_model: Type[llmflex.Embeddings.base_embeddings.BaseEmbeddings], text_splitter: Type[llmflex.TextSplitters.base_text_splitter.BaseTextSplitter], name: str, type: str, embedding_size: int, max_seq_length: int)`
:   Base class for storing the embedding model and the text splitter.

    ### Descendants

    * llmflex.Embeddings.api_embeddings.APIEmbeddingsToolkit
    * llmflex.Embeddings.huggingface_embeddings.HuggingfaceEmbeddingsToolkit

    ### Instance variables

    `embedding_model: llmflex.Embeddings.base_embeddings.BaseEmbeddings`
    :   The embedding model.
        
        Returns:
            BaseEmbeddings: The embedding model.

    `embedding_size: int`
    :   The embedding model's output dimensions.
        
        Returns:
            int: The embedding model's output dimensions.

    `langchain_embeddings: llmflex.Embeddings.base_embeddings.LangchainEmbeddings`
    :   Langchain compatible embeddings model.
        
        Returns:
            LangchainEmbeddings: Langchain compatible embeddings model.

    `max_seq_length: int`
    :   Maximum number of tokens used in each embedding vector.
        
        Returns:
            int: Maximum number of tokens used in each embedding vector.

    `name: str`
    :   Name of the embedding model.
        
        Returns:
            str: Name of the embedding model.

    `text_splitter: llmflex.TextSplitters.base_text_splitter.BaseTextSplitter`
    :   The text splitter.
        
        Returns:
            BaseTextSplitter: The text splitter.

    `type: str`
    :   Type of the embedding toolkit.
        
        Returns:
            str: Type of the embedding toolkit.

    ### Methods

    `batch_embed(self, texts: List[str]) ‑> numpy.ndarray[numpy.float32]`
    :   Embed list of texts.
        
        Args:
            texts (List[str]): List of text to embed.
        
        Returns:
            np.ndarray[np.float32]: Array of embedding vectors of the list of texts.

    `embed(self, text: str) ‑> numpy.ndarray[numpy.float32]`
    :   Embed a single string.
        
        Args:
            text (str): String to embed.
        
        Returns:
            np.ndarray[np.float32]: Vector of the embedded stirng.

`LangchainEmbeddings(model: Type[llmflex.Embeddings.base_embeddings.BaseEmbeddings])`
:   Class for langchain compatible embeddings.

    ### Ancestors (in MRO)

    * langchain_core.embeddings.Embeddings
    * abc.ABC

    ### Methods

    `embed_documents(self, texts: List[str]) ‑> List[List[float]]`
    :   Embed search docs.

    `embed_query(self, text: str) ‑> List[float]`
    :   Embed query text.