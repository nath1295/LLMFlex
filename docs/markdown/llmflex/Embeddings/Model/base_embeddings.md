Module llmflex.Embeddings.Model.base_embeddings
===============================================

Classes
-------

`BaseEmbeddings(model: Any, model_id: str, tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer, max_seq_len: int, embedding_size: int)`
:   Base class for embedding models.
        
    
    Initializes the base embedding model.
    
    This method sets up the base embedding model with the given parameters.
    Args:
        model (Any): The underlying model for the embeddings.
        model_id (str): The ID of the embeddings model.
        tokenizer (BaseTokenizer): The tokenizer used to convert text into tokens.
        max_seq_len (int): The maximum sequence length that the embeddings can handle.
        embedding_size (int): The size of the embeddings.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.Embeddings.Model.api_embeddings.APIEmbeddings
    * llmflex.Embeddings.Model.huggingface_embeddings.HuggingFaceEmbeddings

    ### Instance variables

    `embedding_size: int`
    :   Gets the size of the embeddings.
        
        Returns:
            int: The size of the embeddings.

    `max_seq_len: int`
    :   Gets the maximum sequence length that the embeddings can handle.
        
        Returns:
            int: The maximum sequence length.

    `model_id: str`
    :   Gets the ID of the embeddings model.
        
        Returns:
            str: The ID of the embeddings model.

    `tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer`
    :   Gets the tokenizer used to convert text into tokens.
        
        Returns:
            BaseTokenizer: The tokenizer used to convert text into tokens.

    ### Methods

    `batch_embed(self, texts: List[str]) ‑> numpy.ndarray`
    :   Embeds a batch of text sequences.
        
        This method converts a batch of text sequences into their corresponding embeddings using the underlying model and tokenizer.
        Args:
            texts (List[str]): A sequence of text strings to embed.
        
        Returns:
            np.ndarray: A 2D numpy array of shape (len(text), embedding_size) containing the embeddings for each text string in the input sequence.

    `embed(self, text: str) ‑> numpy.ndarray`
    :   Embeds a single text sequence.
        
        This method converts a single text string into its corresponding embedding using the underlying model and tokenizer.
        Args:
            text (str): A single text string to embed.
        
        Returns:
            np.ndarray: A 1D numpy array of shape (embedding_size,) containing the embedding for the input text string.