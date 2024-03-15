Module llmflex.VectorDBs.base_vectordb
======================================

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

Classes
-------

`BaseVectorDatabase(embeddings: Type[BaseEmbeddingsToolkit], name: Optional[str] = None, vectordb_dir: Optional[str] = None)`
:   Base class for vector databases.
        
    
    Initialise a vector database.
    
    Args:
        embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit to use.
        name (Optional[str], optional): Name of the vector database. Will be used as the directory base name of the vector database in vectordb_dir. If None is given, the vector database will not be saved. Defaults to None.
        vectordb_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.VectorDBs.faiss_vectordb.FaissVectorDatabase

    ### Static methods

    `from_documents(embeddings: Type[BaseEmbeddingsToolkit], docs: List[Document], name: Optional[str] = None, vectordb_dir: Optional[str] = None, split_text: bool = True, text_splitter: Optional[Type[BaseTextSplitter]] = None) ‑> llmflex.VectorDBs.base_vectordb.BaseVectorDatabase`
    :   Load the vector database from existing documents.
        
        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit to use.
            docs (List[Document]): List of documents to use.
            name (Optional[str], optional): Name of the vector database. Will be used as the directory base name of the vector database in vectordb_dir. If None is given, the vector database will not be saved. Defaults to None.
            vectordb_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.
            split_text (bool, optional): Whether to split the docuements with the embeddings toolkit text splitter. Defaults to True.
            text_splitter (Optional[Type[BaseTextSplitter]], optional): Text splitter to split the documents. If none given, the embeddings toolkit text splitter will be used. Defaults to None.
        
        Returns:
            BaseVectorDatabase: The initialised vector database.

    `from_exist(embeddings: Type[BaseEmbeddingsToolkit], name: str, vectordb_dir: Optional[str] = None) ‑> llmflex.VectorDBs.base_vectordb.BaseVectorDatabase`
    :   Load the vector database from an existing vector database.
        
        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit to use.
            name (str): Name of the existing database.
            vectordbs_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.
        
        Returns:
            BaseVectorDatabase: The initialised vector database.

    `from_texts(embeddings: Type[BaseEmbeddingsToolkit], texts: List[str], metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None, name: Optional[str] = None, vectordb_dir: Optional[str] = None, split_text: bool = True, text_splitter: Optional[Type[BaseTextSplitter]] = None) ‑> llmflex.VectorDBs.base_vectordb.BaseVectorDatabase`
    :   Load the vector database from existing texts.
        
        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit to use.
            texts (List[str]): List of texts to add.
            metadata (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Metadata to add along with the texts. Defaults to None.
            name (Optional[str], optional): Name of the vector database. Will be used as the directory base name of the vector database in vectordb_dir. If None is given, the vector database will not be saved. Defaults to None.
            vectordb_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.
            split_text (bool, optional): Whether to split the docuements with the embeddings toolkit text splitter. Defaults to True.
            text_splitter (Optional[Type[BaseTextSplitter]], optional): Text splitter to split the documents. If none given, the embeddings toolkit text splitter will be used. Defaults to None.
        
        Returns:
            BaseVectorDatabase: The initialised vector database.

    ### Instance variables

    `data: Dict[int, llmflex.Schemas.documents.Document]`
    :   Dictionary of all the documents in the vector database.
        
        Returns:
            Dict[int, Document]: Dictionary of all the documents in the vector database.

    `db_dir: Optional[str]`
    :   Directory of the vector database.
        
        Returns:
            Optional[str]: Directory of the vector database.

    `embeddings: llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit`
    :   Embeddings toolkit used in the vector database.
        
        Returns:
            BaseEmbeddingsToolkit: Embeddings toolkit used in the vector database.

    `index: Any`
    :   Index of the vector database.
        
        Returns:
            Any: Index of the vector database.

    `info: Dict[str, Any]`
    :   Information of the vector database.
        
        Returns:
            Dict[str, Any]: Information of the vector database.

    `name: Optional[str]`
    :   Name of the vector database.
        
        Returns:
            Optional[str]: Name of the vector database.

    `size: int`
    :   Number of documents in the vector database.
        
        Returns:
            int: Number of documents in the vector database.

    ### Methods

    `add_docs_with_vectors(self, vectors: Sequence[Sequence[float]], docs: List[Document]) ‑> None`
    :   Add documents with pre-embedded vectors into the vector database.
        
        Args:
            vectors (Sequence[Sequence[float]]): Pre-embedded vectors.
            docs (List[Document]): List of documents.

    `add_documents(self, docs: List[Document], split_text: bool = True, text_splitter: Optional[Type[BaseTextSplitter]] = None) ‑> None`
    :   Add documents into the vector database.
        
        Args:
            docs (List[Document]): List of documents to split.
            split_text (bool, optional): Whether to split the docuements with the embeddings toolkit text splitter. Defaults to True.
            text_splitter (Optional[Type[BaseTextSplitter]], optional): Text splitter to split the documents. If none given, the embeddings toolkit text splitter will be used. Defaults to None.

    `add_texts(self, texts: List[str], metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None, split_text: bool = True, text_splitter: Optional[Type[BaseTextSplitter]] = None) ‑> None`
    :   Add texts into the vector database.
        
        Args:
            texts (List[str]): List of texts to add.
            metadata (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Metadata to add along with the texts. Defaults to None.
            split_text (bool, optional): Whether to split the docuements with the embeddings toolkit text splitter. Defaults to True.
            text_splitter (Optional[Type[BaseTextSplitter]], optional): Text splitter to split the documents. If none given, the embeddings toolkit text splitter will be used. Defaults to None.

    `batch_search(self, queries: List[str], top_k: int = 5, index_only: bool = True, batch_size: int = 100, filter_fn: Optional[Callable[[Document], bool]] = None, **kwargs) ‑> List[List[Union[str, Dict[str, Any]]]]`
    :   Batch simlarity search on multiple queries.
        
        Args:
            queries (List[str]): List of queries.
            top_k (int, optional): Maximum number of results for each query. Defaults to 5.
            index_only (bool, optional): Whether to return the list of indexes only. Defaults to True.
            batch_size (int, optional): Batch size to perform similarity search. Defaults to 100.
            filter_fn (Optional[Callable[[Document], bool]], optional): The filter function to limit the scope of similarity search. Defaults to None.
        
        Returns:
            List[List[Union[str, Dict[str, Any]]]]: List of list of search results.

    `clear(self) ‑> None`
    :   Clear the entire vector database. Use it with caution.

    `delete_by_metadata(self, filter_fn: Optional[Callable[[Document], bool]] = None, **kwargs) ‑> None`
    :   Remove records by metadata. Pass the filters on metadata as keyword arguments or pass a filter_fn.
        
        Args:
            filter_fn (Optional[Callable[[Document], bool]], optional): The filter function. Defaults to None.

    `save(self) ‑> None`
    :   Save the vector database.

    `search(self, query: str, top_k: int = 5, index_only: bool = True, filter_fn: Optional[Callable[[Document], bool]] = None, **kwargs) ‑> List[Union[str, Dict[str, Any]]]`
    :   Simlarity search on the given query.
        
        Args:
            query (str): Query for similarity search.
            top_k (int, optional): Maximum number of results. Defaults to 5.
            index_only (bool, optional): Whether to return the list of indexes only. Defaults to True.
            filter_fn (Optional[Callable[[Document], bool]], optional): The filter function to limit the scope of similarity search. Defaults to None.
        
        Returns:
            List[Union[str, Dict[str, Any]]]: List of search results.

    `search_by_metadata(self, ids_only: bool = False, filter_fn: Optional[Callable[[Document], bool]] = None, **kwargs) ‑> Union[List[int], Dict[int, llmflex.Schemas.documents.Document]]`
    :   Search documents or ids by metadata. Pass the filters on metadata as keyword arguments or pass a filter_fn.
        
        Args:
            ids_only (bool, optional): Whether to return a list of ids or a dictionary with the ids as keys and documents as values. Defaults to False.
            filter_fn (Optional[Callable[[Document], bool]], optional): The filter function. Defaults to None.
        
        Returns:
            Union[List[int], Dict[int, Document]]: List of ids or dictionary with the ids as keys and documents as values.