Module llmplus.Embeddings.hf_embeddings_server
==============================================

Classes
-------

`HuggingFaceEmbeddingsServer(model_id: str = 'thenlper/gte-small', default_batch_size: int = 128, **kwargs)`
:   Initialising the model server.
    
    Args:
        model_id (str, optional): Huggingface repo id. Defaults to 'thenlper/gte-small'.
        default_batch_size (int, optional): Default batch size for encoding if not specified on the client side. Defaults to 128.

    ### Methods

    `run(self, **kwargs) ‑> None`
    :   Start the server.