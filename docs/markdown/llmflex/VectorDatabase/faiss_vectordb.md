Module llmflex.VectorDatabase.faiss_vectordb
============================================

Classes
-------

`FaissVectorDatabase(embeddings: BaseEmbeddings, text_splitter: Optional[BaseTextSplitter] = None, split_text: bool = True, vdb_dir: Optional[str] = None, init_save: bool = True, **kwargs)`
:   Vector database class using Faiss to power. 
        
    
    Initializes the vector database.
    
    Args:
        embeddings (BaseEmbeddings): The embedding model to use for vectorization.
        text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.
        split_text (bool, optional): Whether to split the input text into chunks using the text splitter by default. Defaults to True.
        vdb_dir (Optional[str], optional): The directory to store the vector database. Defaults to None.
        init_save (bool, optional): Whether to save the vector database while initialising it. Defaults to True.

    ### Ancestors (in MRO)

    * llmflex.VectorDatabase.base_vectordb.BaseVectorDatabase
    * abc.ABC

    ### Static methods

    `from_exist(embeddings: BaseEmbeddings, vdb_dir: str, text_splitter: Optional[BaseTextSplitter] = None, split_text: bool = True, **kwargs) ‑> llmflex.VectorDatabase.faiss_vectordb.FaissVectorDatabase`
    :   Creates a new vector database instance from an existing vector database.
        
        This method is used to load a vector database that has already been saved to disk.
        
        Args:
            embeddings (BaseEmbeddings): The embedding model to use for vectorization.
            vdb_dir (str): The directory path to the existing vector database.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.
            split_text (bool, optional): Whether to split the input text into chunks using the text splitter by default. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the vector database constructor.
            
        Returns:
            FaissVectorDatabase: A new vector database instance loaded from the existing vector database.