Module llmflex.VectorDatabase.base_vectordb
===========================================

Classes
-------

`BaseVectorDatabase(embeddings: BaseEmbeddings, text_splitter: Optional[BaseTextSplitter] = None, split_text: bool = True, vdb_dir: Optional[str] = None, init_save: bool = True, **kwargs)`
:   Base class of vector database.
        
    
    Initializes the base vector database.
    
    Args:
        embeddings (BaseEmbeddings): The embedding model to use for vectorization.
        text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.
        split_text (bool, optional): Whether to split the input text into chunks using the text splitter by default. Defaults to True.
        vdb_dir (Optional[str], optional): The directory to store the vector database. Defaults to None.
        init_save (bool, optional): Whether to save the vector database while initialising it. Defaults to True.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.VectorDatabase.faiss_vectordb.FaissVectorDatabase
    * llmflex.VectorDatabase.numpy_vectordb.NumpyVectorDatabase

    ### Static methods

    `from_exist(embeddings: BaseEmbeddings, vdb_dir: str, text_splitter: Optional[BaseTextSplitter] = None, split_text: bool = True, **kwargs) ‑> llmflex.VectorDatabase.base_vectordb.BaseVectorDatabase`
    :   Creates a new vector database instance from an existing vector database.
        
        This method is used to load a vector database that has already been saved to disk.
        
        Args:
            embeddings (BaseEmbeddings): The embedding model to use for vectorization.
            vdb_dir (str): The directory path to the existing vector database.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.
            split_text (bool, optional): Whether to split the input text into chunks using the text splitter by default. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the vector database constructor.
        
        Returns:
            BaseVectorDatabase: A new vector database instance loaded from the existing vector database.

    ### Instance variables

    `data: np.ndarray`
    :   Gets the list of documents stored in the vector database.
        
        Returns:
            np.ndarray: A list of documents stored in the vector database.

    `doc_ids: np.ndarray`
    :   Gets the unique identifiers of the documents stored in the vector database.
        
        Returns:
            np.ndarray: A 1D NumPy array containing the unique identifiers of the documents.

    `embeddings: BaseEmbeddings`
    :   Gets the embedding model used for vectorization.
        
        Returns:
            BaseEmbeddings: The embedding model.

    `info: Dict[str, Any]`
    :   Returns a dictionary containing information about the vector database.
        
        Returns:
            Dict[str, Any]: A dictionary containing information about the vector database.

    `size: int`
    :   Gets the number of documents stored in the vector database.
        
        Returns:
            int: The number of documents stored in the vector database.

    `split_text: bool`
    :   Gets whether the input text is split into chunks using the text splitter by default.
        
        Returns:
            bool: True if the input text is split by default, False otherwise.

    `text_splitter: BaseTextSplitter`
    :   Gets the text splitter used for splitting input text into chunks.
        
        Returns:
            BaseTextSplitter: The text splitter.

    `vdb_dir: Optional[str]`
    :   Gets the directory to store the vector database.
        
        Returns:
            Optional[str]: The directory path, or None if not set.

    `vectors: np.ndarray`
    :   Gets the vectors of the documents stored in the vector database.
        
        Returns:
            np.ndarray: A 2D NumPy array containing the vectors of the documents, where each row represents a document vector.

    ### Methods

    `add_documents(self, docs: Union[Document, List[Document]], split_text: Optional[bool] = None, text_splitter: Optional[BaseTextSplitter] = None) ‑> None`
    :   Adds new documents to the vector database.
        
        If `split_text` is True or not provided, the input documents will be split into chunks using the provided `text_splitter` before being added to the vector database.
        
        Args:
            docs (Union[Document, List[Document]]): The document(s) to add to the vector database. Can be a single `Document` object or a list of `Document` objects.
            split_text (Optional[bool], optional): Whether to split the input text into chunks using the text splitter. Defaults to None.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.

    `add_texts(self, texts: Union[str, List[str]], metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None, split_text: Optional[bool] = None, text_splitter: Optional[BaseTextSplitter] = None) ‑> None`
    :   Adds new texts to the vector database.
        
        If `split_text` is True or not provided, the input texts will be split into chunks using the provided `text_splitter` before being added to the vector database.
        
        Args:
            texts (Union[str, List[str]]): The text(s) to add to the vector database. Can be a single string or a list of strings.
            metadata (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): The metadata to associate with the new documents. Can be a single dictionary or a list of dictionaries. Defaults to None.
            split_text (Optional[bool], optional): Whether to split the input text into chunks using the text splitter. Defaults to None.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.
        
        Raises:
            Exception: Number of texts and number of metadata mismatch

    `batch_search(self, queries: List[str], top_k: int = 5, scope_ids: Optional[List[int]] = None, batch_size: int = 256, **kwargs) ‑> List[List[llmflex.VectorDatabase.base_vectordb.SearchResult]]`
    :   Performs a batch search in the vector database using the provided queries.
        
        This method finds the top `top_k` most similar documents for each input query in the batch.
        
        Args:
            queries (List[str]): A list of input queries for which to perform the search.
            top_k (int, optional): The number of most similar documents to return for each input query. Defaults to 5.
            scope_ids (Optional[List[int]], optional): A list of unique identifiers of the documents to consider for the search. If provided, only documents with these identifiers will be considered. Defaults to None.
            batch_size (int, optional): The number of queries to process at a time. Defaults to 256.
        
        Raises:
            ValueError: If the number of queries is zero.
        
        Returns:
            List[List[SearchResult]]: A list of lists, where each inner list contains the top `top_k` most similar documents for the corresponding input query.

    `clear(self) ‑> None`
    :   Removes all documents from the vector database.
        
        This method removes all documents from the vector database, effectively clearing it.

    `drop_ids(self, doc_ids: List[int]) ‑> None`
    :   Removes documents from the vector database based on their IDs.
        
        This method removes the documents with the provided IDs from the vector database.
        
        Args:
            doc_ids (List[int]): A list of document IDs to remove from the vector database.

    `export_to_dir(self, export_dir: str) ‑> None`
    :   Save the vector database in the given directory.
        
        Args:
            export_dir (str): Directory to save.

    `get_vectors_by_ids(self, doc_ids: Union[List[int], np.ndarray]) ‑> numpy.ndarray`
    :   Retrieves the vectors of the documents with the specified unique identifiers from the vector database.
        
        Args:
            doc_ids (Union[List[int], np.ndarray]): A list or NumPy array containing the unique identifiers of the documents for which to retrieve the vectors.
            
        Returns:
            np.ndarray: A 2D NumPy array containing the vectors of the specified documents, where each row represents a document vector.

    `save(self) ‑> None`
    :   Save everything of the vector database.

    `search(self, query: str, top_k: int = 5, scope_ids: Optional[List[int]] = None, **kwargs) ‑> List[llmflex.VectorDatabase.base_vectordb.SearchResult]`
    :   Performs a search in the vector database using the provided query.
        
        This method finds the top `top_k` most similar documents for the input query.
        
        Args:
            query (str): The input query for which to perform the search.
            top_k (int, optional): The number of most similar documents to return for the input query. Defaults to 5.
            scope_ids (Optional[List[int]], optional): A list of unique identifiers of the documents to consider for the search. If provided, only documents with these identifiers will be considered. Defaults to None.
        
        Returns:
            List[SearchResult]: A list of the top `top_k` most similar documents for the input query.

    `search_by_doc(self, filter_fn: Optional[Callable[[Document], bool]] = None, ids_only: bool = True, **kwargs) ‑> List[int] | Dict[int, llmflex.Schema.document.Document]`
    :   Searches for documents in the vector database based on a filter function.
        
        This method finds documents that satisfy the provided filter function. If `ids_only` is True, it returns a list of document IDs that match the filter. Otherwise, it returns a dictionary mapping document IDs to their corresponding documents.
        
        Args:
            filter_fn (Optional[Callable[[Document], bool]], optional): A function that takes a `Document` object and returns a boolean indicating whether the document matches the filter. Defaults to None.
            ids_only (bool, optional): Whether to return only the IDs of the matching documents or the documents themselves. Defaults to True.
        
        Returns:
            Union[List[int], Dict[int, Document]]: A list of document IDs that match the filter if `ids_only` is True, or a dictionary mapping document IDs to their corresponding documents otherwise.

`SearchResult(**data: Any)`
:   Usage docs: https://docs.pydantic.dev/2.7/concepts/models/
    
    A base class for creating Pydantic models.
    
    Attributes:
        __class_vars__: The names of classvars defined on the model.
        __private_attributes__: Metadata about the private attributes of the model.
        __signature__: The signature for instantiating the model.
    
        __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
        __pydantic_core_schema__: The pydantic-core schema used to build the SchemaValidator and SchemaSerializer.
        __pydantic_custom_init__: Whether the model has a custom `__init__` function.
        __pydantic_decorators__: Metadata containing the decorators defined on the model.
            This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
        __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
            __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
        __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
        __pydantic_post_init__: The name of the post-init method for the model, if defined.
        __pydantic_root_model__: Whether the model is a `RootModel`.
        __pydantic_serializer__: The pydantic-core SchemaSerializer used to dump instances of the model.
        __pydantic_validator__: The pydantic-core SchemaValidator used to validate instances of the model.
    
        __pydantic_extra__: An instance attribute with the values of extra fields from validation when
            `model_config['extra'] == 'allow'`.
        __pydantic_fields_set__: An instance attribute with the names of fields explicitly set.
        __pydantic_private__: Instance attribute with the values of private attributes set on the model instance.
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel

    ### Class variables

    `doc: llmflex.Schema.document.Document`
    :

    `doc_id: int`
    :

    `model_computed_fields`
    :

    `model_config`
    :

    `model_fields`
    :

    `score: float`
    :