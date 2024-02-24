Module llmflex.Data.vector_database
===================================

Functions
---------

    
`default_vectordb_dir() ‑> str`
:   Default home directory of vector databases.
    
    Returns:
        str: Default home directory of vector databases.

    
`list_vectordbs(vectordb_dir: Optional[str] = None) ‑> List[str]`
:   List all the vector databases in the given directory.
    
    Args:
        vectordb_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.
    
    Returns:
        List[str]: List all the vector databases in the given directory.

    
`name_checker(name: str) ‑> str`
:   Raise error if the given string has space, newline characters, or tab characters.
    
    Args:
        name (str): String to check.
    
    Returns:
        str: Return the given text if it passes all the checkes.

    
`texts_to_documents(texts: List[str], embeddings: Optional[Type[BaseEmbeddingsToolkit]] = None, text_splitter: Optional[Type[TextSplitter]] = None, data: Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]] = None, split_text: bool = True) ‑> List[langchain_core.documents.base.Document]`
:   Create splitted documents from the list of text strings.
    
    Args:
        texts (List[str]): List of strings to split into documents.
        embeddings (Optional[Type[BaseEmbeddingsToolkit]], optional): Embedding toolkit used to split the documents. Defaults to None
        text_splitter (Optional[Type[TextSplitter]], optional): Text splitter used to split the documents. If provided, it will be used instead of the embedding toolkit. Defaults to None.
        data (Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]], optional): Metadata for each text strings. Defaults to None.
        split_text (bool, optional): Whether to split text if the given text is too long. Defaults to True.
    
    Returns:
        List[Document]: List of splitted documents.

Classes
-------

`VectorDatabase(embeddings: Type[BaseEmbeddingsToolkit], vectordb_dir: Optional[str] = None, name: Optional[str] = None, save_raw: bool = False)`
:   Vector database class, suitable for storing text data as embeddings for similarity searches and other classes that requires numerical respresentations of texts.
        
    
    Initialising basic information of the vector database.
    
    Args:
        embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkits used in the vector database.
        vectordb_dir (Optional[str], optional): Parent directory of the vector database if it is not In-memory only. If None is given, the default_vectordb_dir will be used. Defaults to None.
        name (Optional[str], optional): Name of the vector database. If given, the vector database will be stored in storage. Defaults to None.
        save_raw (bool, optional): Whether to save raw text data and metadata as a separate json file. Defaults to False.

    ### Static methods

    `from_data(index: List[str], embeddings: Type[BaseEmbeddingsToolkit], data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]] = {}, text_splitter: Type[TextSplitter] = None, name: Optional[str] = None, vectordb_dir: Optional[str] = None, save_raw: bool = False, split_text: bool = True) ‑> llmflex.Data.vector_database.VectorDatabase`
    :   Initialise the vector database with list of texts.
        
        Args:
            index (List[str]): List of texts to initialise the database.
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkits used in the vector database.
            data (Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]], optional): Metadata for the list of texts. Defaults to dict().
            text_splitter (Optional[Type[TextSplitter]], optional): Text splitter used to split the documents. If provided, it will be used instead of the embedding toolkit. Defaults to None.
            name (Optional[str], optional): Name of the vector database. If given, the vector database will be stored in storage. Defaults to None.
            vectordb_dir (Optional[str], optional): Parent directory of the vector database if it is not In-memory only. If None is given, the default_vectordb_dir will be used. Defaults to None.
            save_raw (bool, optional): Whether to save raw text data and metadata as a separate json file. Defaults to False.
            split_text (bool, optional): Whether to split the texts if they are too long. Defaults to True.
        
        Returns:
            VectorDatabase: The intialised vector database.

    `from_empty(embeddings: Type[BaseEmbeddingsToolkit], name: Optional[str] = None, vectordb_dir: Optional[str] = None, save_raw: bool = False) ‑> llmflex.Data.vector_database.VectorDatabase`
    :   Initialise an empty vector database.
        
        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkits used in the vector database.
            name (Optional[str], optional): Name of the vector database. If given, the vector database will be stored in storage. Defaults to None.
            vectordb_dir (Optional[str], optional): Parent directory of the vector database if it is not In-memory only. If None is given, the default_vectordb_dir will be used. Defaults to None.
            save_raw (bool, optional): Whether to save raw text data and metadata as a separate json file. Defaults to False.
        
        Returns:
            VectorDatabase: The intialised vector database.

    `from_exist(name: str, embeddings: Type[BaseEmbeddingsToolkit], vectordb_dir: Optional[str] = None) ‑> llmflex.Data.vector_database.VectorDatabase`
    :   Initialise the vector database from existing files.
        
        Args:
            name (str): Name of the existing vector database.
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit used in this vector database.
            vectordb_dir (Optional[str], optional): Parent directory of the vector database. If None is given, the default_vectordb_dir will be used. Defaults to None.
        
        Returns:
            VectorDatabase: The intialised vector database.

    ### Instance variables

    `data: List[Dict[str, Any]]`
    :   Raw data of the vector database.
        
        Returns:
            List[Dict[str, Any]]: Raw data of the vector database.

    `embeddings: Type[llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit]`
    :   Embeddings toolkit used in the vector database.
        
        Returns:
            Type[BaseEmbeddingsToolkit]: Embeddings toolkit used in the vector database.

    `info: Dict[str, Any]`
    :   Information of the vector database.
        
        Returns:
            Dict[str, Any]: Information of the vector database.

    `name: str`
    :   Name of the vector database.
        
        Returns:
            str: Name of the vector database.

    `save_raw: bool`
    :   Whether to save the raw data as json or not.
        
        Returns:
            bool: Whether to save the raw data as json or not.

    `size: int`
    :   Number of embeddings in the vector database. May be more than the number of texts you have added into the database due to text splitting for longer texts.
        
        Returns:
            int: Number of embeddings.

    `vdb_dir: Optional[str]`
    :   Directory of the vector database if it is not in-memory only.
        
        Returns:
            Union[str, None]: Directory of the vector database if it is not in-memory only.

    `vectorstore: 'FAISS'`
    :   Return the faiss vectorstore
        
        Returns:
            FAISS: The vector store.

    ### Methods

    `add_documents(self, docs: List[Document], text_splitter: Optional[Type[TextSplitter]] = None, split_text: bool = True) ‑> None`
    :   Adding documents to the vector database.
        
        Args:
            docs (List[Document]): List of documents to add.
            text_splitter (Optional[Type[TextSplitter]], optional): Text splitter used to split the documents. If provided, it will be used instead of the embedding toolkit. Defaults to None.
            split_text (bool, optional): Whether to split the texts if they are too long. Defaults to True.

    `add_texts(self, texts: List[str], metadata: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]] = {}, text_splitter: Optional[Type[TextSplitter]] = None, split_text: bool = True) ‑> None`
    :   Adding texts to the vector database.
        
        Args:
            texts (List[str]): List of texts to add.
            metadata (Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]], optional): Metadata for the texts. Defaults to dict().
            text_splitter (Optional[Type[TextSplitter]], optional): Text splitter used to split the documents. If provided, it will be used instead of the embedding toolkit. Defaults to None.
            split_text (bool, optional): Whether to split the texts if they are too long. Defaults to True.

    `delete_by_metadata(self, **kwargs: Dict[str, Any]) ‑> None`
    :   Remove records base on the given key value pairs criteria.
        
        Raises:
            ValueError: If not key value pairs given, this error will be raised. To clear the whole vector database, please use the "clear()" method.

    `save(self) ‑> None`
    :   Save the latest vector database.

    `search(self, query: str, top_k: int = 5, index_only: bool = True, **kwargs) ‑> List[Union[str, Dict[str, Any]]]`
    :   Similarity search of text on the vector database. Pass keyword arguments as filters on metadata.
        
        Args:
            query (str): Text search string query.
            top_k (int, optional): Maximum number of results to return. Defaults to 5.
            index_only (bool, optional): If set as True, only the index string will be returned. Otherwise, metadata and scores will be returned as well. Defaults to True.
        
        Returns:
            List[Union[str, Dict[str, Any]]]: List of search results.

    `search_by_metadata(self, **kwargs: Dict[str, Any]) ‑> Dict[str, langchain_core.documents.base.Document]`
    :   Exact match search on metadata. Filters should be provided as key value pair arguments.
        
        Returns:
            Dict[str, Document]: Dictionary of saerch results, with docstore ids as the keys.