Module llmflex.Embeddings.Model.api_embeddings
==============================================

Classes
-------

`APIEmbeddings(base_url: str, batch_size: int = 256, verbose: bool = False, normalize: bool = True, tokenizer_name_or_path: str | None = None, tokenizer_kwargs: Dict[str, Any] | None = None)`
:   A class for handling embeddings using an API.
    
    This class extends the BaseEmbeddings class and provides methods for generating embeddings using an API.
    
    Initializes the APIEmbeddings instance.
    Args:
        base_url (str): The base URL of the API to use for generating embeddings.
        batch_size (int, optional): The maximum number of items to send in a single API request. Defaults to 256.
        verbose (bool, optional): Whether to print debug information. Defaults to False.
        normalize (bool, optional): Whether to normalize the generated embeddings. Defaults to True.
        tokenizer_name_or_path (Optional[str], optional): The name or path of the tokenizer to use for tokenizing input text. Defaults to None.
        tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Embeddings.Model.base_embeddings.BaseEmbeddings
    * abc.ABC