Module llmflex.Embeddings.Model.huggingface_embeddings
======================================================

Classes
-------

`HuggingFaceEmbeddings(pretrained_model_name_or_path: str, batch_size: int = 256, verbose: bool = False, normalize: bool = True, max_seq_len: int | None = None, model_kwargs: Dict[str, Any] | None = None, tokenizer_kwargs: Dict[str, Any] | None = None)`
:   A class for handling embeddings using Hugging Face models.
    
    This class inherits from the BaseEmbeddings class and provides methods for
    converting tokens to embeddings using Hugging Face models. It uses the
    HuggingFaceTokenizer class for tokenization and PyTorch for tensor operations.
    
    Initializes the HuggingFaceEmbeddings class.
    
    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model to use.
        batch_size (int, optional): The batch size to use for tokenization and embedding generation. Defaults to 256.
        verbose (bool, optional): Whether to print verbose output during tokenization and embedding generation. Defaults to False.
        normalize (bool, optional): Whether to normalize the embeddings. Defaults to True.
        max_seq_len (Optional[int], optional): The maximum sequence length to use for tokenization and embedding generation. Defaults to None.
        model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the Hugging Face model. Defaults to None.
        tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the Hugging Face tokenizer. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Embeddings.Model.base_embeddings.BaseEmbeddings
    * abc.ABC