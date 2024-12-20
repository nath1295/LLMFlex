Module llmflex.Embeddings.hf_embedding_server
=============================================

Classes
-------

`HFEmbeddingServer(pretrained_model_name_or_path: str, batch_size: int = 256, normalize: bool = True, model_kwargs: Dict[str, Any] | None = None, tokenizer_kwargs: Dict[str, Any] | None = None)`
:   Class for serving a HuggingFace embedding model on an API server.
        
    
    Initialising the server class.
    
    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model to use.
        batch_size (int, optional): The batch size to use for tokenization and embedding generation. Defaults to 256.
        normalize (bool, optional): Whether to normalize the embeddings. Defaults to True.
        model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the Hugging Face model. Defaults to None.
        tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the Hugging Face tokenizer. Defaults to None.

    ### Methods

    `run(self, **kwargs) ‑> None`
    :   Start the server.