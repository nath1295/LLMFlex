Module llmflex.Embeddings.base_embeddings
=========================================

Classes
-------

`BaseEmbeddingsToolkit()`
:   Base class for storing the embedding model and the text splitter.
        
    
    Initialising the embedding toolkit.

    ### Descendants

    * llmflex.Embeddings.api_embeddings.APIEmbeddingsToolkit
    * llmflex.Embeddings.huggingface_embeddings.HuggingfaceEmbeddingsToolkit

    ### Instance variables

    `embedding_model: Type[langchain_core.embeddings.Embeddings]`
    :   The embedding model.
        
        Returns:
            Embeddings: The embedding model.

    `embedding_size: int`
    :   The embedding model's output dimensions.
        
        Returns:
            int: The embedding model's output dimensions.

    `name: str`
    :   Name of the embedding model.
        
        Returns:
            str: Name of the embedding model.

    `text_splitter: Type[langchain.text_splitter.TextSplitter]`
    :   The text splitter.
        
        Returns:
            TextSplitter: The text splitter.

    `type: str`
    :   Type of the embedding toolkit.
        
        Returns:
            str: Type of the embedding toolkit.