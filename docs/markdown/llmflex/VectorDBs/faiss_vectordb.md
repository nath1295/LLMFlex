Module llmflex.VectorDBs.faiss_vectordb
=======================================

Classes
-------

`FaissVectorDatabase(embeddings: Type[BaseEmbeddingsToolkit], name: Optional[str] = None, vectordb_dir: Optional[str] = None)`
:   Base class for vector databases.
        
    
    Initialise a vector database.
    
    Args:
        embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit to use.
        name (Optional[str], optional): Name of the vector database. Will be used as the directory base name of the vector database in vectordb_dir. If None is given, the vector database will not be saved. Defaults to None.
        vectordb_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.VectorDBs.base_vectordb.BaseVectorDatabase
    * abc.ABC